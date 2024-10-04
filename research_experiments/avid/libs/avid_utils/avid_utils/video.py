import torch.nn.functional as F
from einops import rearrange


def resize_video(video_tensor, new_size):
    """
    Resize a tensor of shape (b, t, h, w, c) to (b, t, new_h, new_w, c).

    Parameters:
    tensor (torch.Tensor): Input tensor of shape (b, t, h, w, c)
    new_size (tuple): New size (new_h, new_w)

    Returns:
    torch.Tensor: Resized tensor of shape (b, t, new_h, new_w, c)
    """
    original_dtype = video_tensor.dtype
    video_tensor = video_tensor.float()
    b, t, _, _, _ = video_tensor.shape
    video_tensor = rearrange(video_tensor, "b t h w c -> (b t) c h w")

    # Apply the resizing transform using interpolate
    resized_tensor = F.interpolate(video_tensor, size=new_size, mode="bilinear", align_corners=False)
    resized_tensor = rearrange(resized_tensor, "(b t) c h w -> b t h w c", b=b, t=t)
    resized_tensor = resized_tensor.to(original_dtype)
    return resized_tensor
