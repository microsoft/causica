import torch
import torch.nn as nn
from avid_utils.helpers import prob_mask_like
from einops import rearrange
from lvdm.basics import avg_pool_nd, conv_nd, linear, zero_module
from lvdm.models.utils_diffusion import timestep_embedding
from lvdm.modules.attention import SpatialTransformer, TemporalTransformer
from lvdm.modules.networks.openaimodel3d import (
    Downsample,
    DownsampleToSize,
    ResBlock,
    TimestepEmbedSequential,
    UNetModel,
    UpsampleToSize,
)


class ControlNet(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        original_size=[40, 64],
        output_mask=False,
        action_conditioned=False,
        action_dropout_prob=0.0,
        action_dims=7,
        downsample_size=None,
        conv_resample=True,
        dims=2,
        context_dim=None,
        use_scale_shift_norm=False,
        resblock_updown=False,
        num_heads=-1,
        num_head_channels=-1,
        transformer_depth=1,
        use_linear=False,
        use_checkpoint=False,
        temporal_conv=False,
        tempspatial_aware=False,
        temporal_attention=True,
        use_relative_position=True,
        use_causal_attention=False,
        temporal_length=None,
        use_fp16=False,
        addition_attention=False,
        temporal_selfatt_only=True,
        image_cross_attention=False,
        image_cross_attention_scale_learnable=False,
        default_fs=4,
        fs_condition=False,
    ):
        super(ControlNet, self).__init__()
        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"
        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.temporal_attention = temporal_attention
        embed_dim = model_channels * 4
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        temporal_self_att_only = True
        self.addition_attention = addition_attention
        self.temporal_length = temporal_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
        self.default_fs = default_fs
        self.fs_condition = fs_condition
        self.output_mask = output_mask

        ## Downsample inputs and upsample outputs
        self.downsample_size = downsample_size
        if downsample_size:
            self.input_downsample = DownsampleToSize(self.downsample_size)
            self.output_upsample = UpsampleToSize(original_size)

        ## if action conditioned, add action embedding
        self.action_conditioned = action_conditioned
        if self.action_conditioned:
            self.action_embed = nn.Sequential(
                linear(action_dims, embed_dim),
                nn.SiLU(),
                linear(embed_dim, embed_dim // 2),
            )
            self.zero_linear_action_mlp = zero_module(nn.Linear(embed_dim // 2, embed_dim // 2))
            time_embed_dim = embed_dim // 2
            self.null_action_emb = nn.Parameter(torch.zeros(1, embed_dim // 2))
            self.action_dropout_prob = action_dropout_prob
        else:
            time_embed_dim = embed_dim

        ## Time embedding blocks
        self.time_embed = nn.Sequential(
            linear(model_channels, embed_dim),
            nn.SiLU(),
            linear(embed_dim, time_embed_dim),
        )
        if fs_condition:
            self.fps_embedding = nn.Sequential(
                linear(model_channels, embed_dim),
                nn.SiLU(),
                linear(embed_dim, embed_dim),
            )
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)
        ## Input Block
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        # Even the input is zero conv'd?
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])
        if self.addition_attention:
            self.init_attn = TimestepEmbedSequential(
                TemporalTransformer(
                    model_channels,
                    n_heads=8,
                    d_head=num_head_channels,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    use_checkpoint=use_checkpoint,
                    only_self_att=temporal_selfatt_only,
                    causal_attention=False,
                    relative_position=use_relative_position,
                    temporal_length=temporal_length,
                )
            )

        self.input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        tempspatial_aware=tempspatial_aware,
                        use_temporal_conv=temporal_conv,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_linear=use_linear,
                            use_checkpoint=use_checkpoint,
                            disable_self_attn=False,
                            video_length=temporal_length,
                            image_cross_attention=self.image_cross_attention,
                            image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable,
                        )
                    )
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                use_linear=use_linear,
                                use_checkpoint=use_checkpoint,
                                only_self_att=temporal_self_att_only,
                                causal_attention=use_causal_attention,
                                relative_position=use_relative_position,
                                temporal_length=temporal_length,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self.input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                self.input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        layers = [
            ResBlock(
                ch,
                embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                tempspatial_aware=tempspatial_aware,
                use_temporal_conv=temporal_conv,
            ),
            SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                use_linear=use_linear,
                use_checkpoint=use_checkpoint,
                disable_self_attn=False,
                video_length=temporal_length,
                image_cross_attention=self.image_cross_attention,
                image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable,
            ),
        ]
        if self.temporal_attention:
            layers.append(
                TemporalTransformer(
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    use_linear=use_linear,
                    use_checkpoint=use_checkpoint,
                    only_self_att=temporal_self_att_only,
                    causal_attention=use_causal_attention,
                    relative_position=use_relative_position,
                    temporal_length=temporal_length,
                )
            )
        layers.append(
            ResBlock(
                ch,
                embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                tempspatial_aware=tempspatial_aware,
                use_temporal_conv=temporal_conv,
            )
        )

        ## Middle Block
        self.middle_block = TimestepEmbedSequential(*layers)
        self.middle_block_out = self.make_zero_conv(ch)

    def make_zero_conv(self, channels):
        # Assuming here that we will always have conv3d, we can pass another dim instead of 3
        return TimestepEmbedSequential(zero_module(conv_nd(3, channels, channels, 1, padding=0)))

    def forward(
        self, x, timesteps, context=None, act=None, dropout_actions=True, features_adapter=None, fs=None, **kwargs
    ):
        b, _, t, _, _ = x.shape
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).type(x.dtype)

        ## repeat t times for context [(b t) 77 768] & time embedding
        ## check if we use per-frame image conditioning
        _, l_context, _ = context.shape
        if l_context == 77 + t * 16:  ## !!! HARD CODE here
            context_text, context_img = context[:, :77, :], context[:, 77:, :]
            context_text = context_text.repeat_interleave(repeats=t, dim=0)
            context_img = rearrange(context_img, "b (t l) c -> (b t) l c", t=t)
            context = torch.cat([context_text, context_img], dim=1)
        else:
            context = context.repeat_interleave(repeats=t, dim=0)

        ## action conditioning
        assert self.action_conditioned, "Control net model requires actions"
        if not self.action_conditioned:
            emb = self.time_embed(t_emb)
            emb = emb.repeat_interleave(repeats=t, dim=0)
        else:
            act_drop_prob = self.action_dropout_prob if dropout_actions else 0.0
            time_emb = self.time_embed(t_emb)
            time_emb = time_emb.repeat_interleave(repeats=t, dim=0)

            # if actions are provided embed them, apply random dropout, concat with time embedding
            if act is not None:
                act_emb = self.action_embed(act)
                keep_mask = prob_mask_like((act.shape[0],), 1 - act_drop_prob, device=act.device)
                act_emb = torch.where(rearrange(keep_mask, "b -> b 1 1"), act_emb, self.null_action_emb)
                act_emb = rearrange(act_emb, "b t c -> (b t) c")
            else:
                act_emb = self.null_action_emb.repeat_interleave(repeats=t * b, dim=0)
            act_emb = self.zero_linear_action_mlp(act_emb)
            emb = torch.cat([time_emb, act_emb], dim=1)

        ## always in shape (b t) c h w, except for temporal layer
        x = rearrange(x, "b c t h w -> (b t) c h w")

        ## combine emb
        if self.fs_condition:
            if fs is None:
                fs = torch.tensor([self.default_fs] * b, dtype=torch.long, device=x.device)
            fs_emb = timestep_embedding(fs, self.model_channels, repeat_only=False).type(x.dtype)

            fs_embed = self.fps_embedding(fs_emb)
            fs_embed = fs_embed.repeat_interleave(repeats=t, dim=0)
            emb = emb + fs_embed

        outs = []
        h = x.type(self.dtype)
        adapter_idx = 0
        for (id, module), zero_conv in zip(enumerate(self.input_blocks), self.zero_convs):
            h = module(h, emb, context=context, batch_size=b)
            if id == 0 and self.addition_attention:
                h = self.init_attn(h, emb, context=context, batch_size=b)
            ## plug-in adapter features
            if ((id + 1) % 3 == 0) and features_adapter is not None:
                h = h + features_adapter[adapter_idx]
                adapter_idx += 1
            outs.append(zero_conv(h, emb, context, batch_size=b))
        if features_adapter is not None:
            assert len(features_adapter) == adapter_idx, "Wrong features_adapter"

        h = self.middle_block(h, emb, context=context, batch_size=b)
        outs.append(self.middle_block_out(h, emb, context, batch_size=b))
        return outs


class ControlNetSmaller(ControlNet):
    def __init__(
        self,
        in_channels,
        model_channels,
        model_channels_new,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        original_size=[40, 64],
        output_mask=False,
        action_conditioned=False,
        action_dropout_prob=0.0,
        action_dims=7,
        downsample_size=None,
        conv_resample=True,
        dims=2,
        context_dim=None,
        use_scale_shift_norm=False,
        resblock_updown=False,
        num_heads=-1,
        num_head_channels=-1,
        transformer_depth=1,
        use_linear=False,
        use_checkpoint=False,
        temporal_conv=False,
        tempspatial_aware=False,
        temporal_attention=True,
        use_relative_position=True,
        use_causal_attention=False,
        temporal_length=None,
        use_fp16=False,
        addition_attention=False,
        temporal_selfatt_only=True,
        image_cross_attention=False,
        image_cross_attention_scale_learnable=False,
        default_fs=4,
        fs_condition=False,
    ):
        super().__init__(
            in_channels,
            model_channels_new,  # changed from model_channels to model_channels_new
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout,
            channel_mult,
            original_size,
            output_mask,
            action_conditioned,
            action_dropout_prob,
            action_dims,
            downsample_size,
            conv_resample,
            dims,
            context_dim,
            use_scale_shift_norm,
            resblock_updown,
            num_heads,
            num_head_channels,
            transformer_depth,
            use_linear,
            use_checkpoint,
            temporal_conv,
            tempspatial_aware,
            temporal_attention,
            use_relative_position,
            use_causal_attention,
            temporal_length,
            use_fp16,
            addition_attention,
            temporal_selfatt_only,
            image_cross_attention,
            image_cross_attention_scale_learnable,
            default_fs,
            fs_condition,
        )
        # dimensions
        self.dim_projections = {}
        dims = [*map(lambda m: model_channels * m, channel_mult)]
        new_dims = [*map(lambda m: model_channels_new * m, channel_mult)]
        for dim, new in zip(dims, new_dims):
            self.dim_projections[new] = dim
        self.projection_layers = nn.ModuleList([])

        for new_dim in self.input_block_chans:
            n_channels_old = self.dim_projections[new_dim]
            n_channels_new = new_dim
            self.projection_layers.append(nn.Linear(n_channels_new, n_channels_old))
        self.projection_layers.append(nn.Linear(new_dim, self.dim_projections[new_dim]))

    def forward(
        self, x, timesteps, context=None, act=None, dropout_actions=True, features_adapter=None, fs=None, **kwargs
    ):
        outs = super().forward(x, timesteps, context, act, dropout_actions, features_adapter, fs)
        new_outs = []
        for i, out in enumerate(outs):
            out_rearrange = rearrange(out, "b c h w -> (b h w) c")
            out_rearrange_output = self.projection_layers[i](out_rearrange)
            out_rearrange_output = rearrange(
                out_rearrange_output,
                "(b h w) c -> b c h w",
                b=out.shape[0],
                h=out.shape[2],
                w=out.shape[3],
            )
            new_outs.append(out_rearrange_output)
        return new_outs


class ControlledUnetModel(UNetModel):
    def forward(
        self,
        x,
        timesteps,
        context=None,
        act=None,
        dropout_actions=True,
        features_adapter=None,
        fs=None,
        control=None,
        only_mid_control=False,
        **kwargs
    ):
        b, _, t, _, _ = x.shape
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).type(x.dtype)

        ## repeat t times for context [(b t) 77 768] & time embedding
        ## check if we use per-frame image conditioning
        _, l_context, _ = context.shape
        if l_context == 77 + t * 16:  ## !!! HARD CODE here
            context_text, context_img = context[:, :77, :], context[:, 77:, :]
            context_text = context_text.repeat_interleave(repeats=t, dim=0)
            context_img = rearrange(context_img, "b (t l) c -> (b t) l c", t=t)
            context = torch.cat([context_text, context_img], dim=1)
        else:
            context = context.repeat_interleave(repeats=t, dim=0)

        ## action conditioning
        assert self.action_conditioned is False, "ControlledUnetModel should not support action conditioning"
        if not self.action_conditioned:
            emb = self.time_embed(t_emb)
            emb = emb.repeat_interleave(repeats=t, dim=0)
        else:
            act_drop_prob = self.action_dropout_prob if dropout_actions else 0.0
            time_emb = self.time_embed(t_emb)
            time_emb = time_emb.repeat_interleave(repeats=t, dim=0)

            # if actions are provided embed them, apply random dropout, concat with time embedding
            if act is not None:
                act_emb = self.action_embed(act)
                keep_mask = prob_mask_like((act.shape[0],), 1 - act_drop_prob, device=act.device)
                act_emb = torch.where(rearrange(keep_mask, "b -> b 1 1"), act_emb, self.null_action_emb)
                act_emb = rearrange(act_emb, "b t c -> (b t) c")
            else:
                act_emb = self.null_action_emb.repeat_interleave(repeats=t * b, dim=0)
            emb = torch.cat([time_emb, act_emb], dim=1)

        ## always in shape (b t) c h w, except for temporal layer
        x = rearrange(x, "b c t h w -> (b t) c h w")

        ## combine emb
        if self.fs_condition:
            if fs is None:
                fs = torch.tensor([self.default_fs] * b, dtype=torch.long, device=x.device)
            fs_emb = timestep_embedding(fs, self.model_channels, repeat_only=False).type(x.dtype)

            fs_embed = self.fps_embedding(fs_emb)
            fs_embed = fs_embed.repeat_interleave(repeats=t, dim=0)
            emb = emb + fs_embed

        h = x.type(self.dtype)
        adapter_idx = 0
        hs = []
        for id, module in enumerate(self.input_blocks):
            h = module(h, emb, context=context, batch_size=b)
            if id == 0 and self.addition_attention:
                h = self.init_attn(h, emb, context=context, batch_size=b)
            ## plug-in adapter features
            if ((id + 1) % 3 == 0) and features_adapter is not None:
                h = h + features_adapter[adapter_idx]
                adapter_idx += 1
            hs.append(h)
        if features_adapter is not None:
            assert len(features_adapter) == adapter_idx, "Wrong features_adapter"

        h = self.middle_block(h, emb, context=context, batch_size=b)
        if control is not None:
            h += control.pop()

        for module in self.output_blocks:
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context=context, batch_size=b)
        h = h.type(x.dtype)
        y = self.out(h)

        # reshape back to (b c t h w)
        y = rearrange(y, "(b t) c h w -> b c t h w", b=b)

        if hasattr(self, "out_mask"):
            mask = self.out_mask(h)
            mask = rearrange(mask, "(b t) c h w -> b c t h w", b=b)
            return y, mask
        return y
