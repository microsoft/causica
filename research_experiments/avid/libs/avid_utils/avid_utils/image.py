import torch
from einops import rearrange


def preprocess_images(images: torch.Tensor) -> torch.Tensor:
    """Preprocess images to be in the range [-1, 1].

    Args:
        images: Tensor of images with values in the range [0, 255] in shape (b, h, w, c)

    Returns:
        Tensor of images with values in the range [-1, 1] in shape (b, c, h, w)
    """
    assert not torch.is_floating_point(images), "Expected images to be of type uint8"
    assert torch.all((images >= 0) & (images <= 255)), "Tensor values must be in the range [0, 255]"
    assert images.ndim in (4, 5), "Expected images to be of shape (b, c, h, w) or (b, t, c, h, w)"

    if images.ndim == 4:
        images = rearrange(images, "b h w c -> b c h w")
    elif images.ndim == 5:
        images = rearrange(images, "b t h w c -> b c t h w")
    images = (images.float() / 255.0) * 2.0 - 1.0
    return images


def revert_preprocess_images(images: torch.Tensor) -> torch.Tensor:
    """Revert the preprocessing of images to be in the range [0, 255].

    Args:
        images: Tensor of images with values in the range [-1, 1] in shape (b, c, h, w)

    Returns:
        Tensor of images with values in the range [0, 255] in shape (b, h, w, c)
    """
    assert torch.is_floating_point(images), "Expected images to be of type float"
    assert images.ndim in (4, 5), "Expected images to be of shape (b, c, h, w) or (b, t, c, h, w)"

    images = (images + 1.0) / 2.0 * 255.0
    images = images.clamp(0, 255).to(torch.uint8)
    if images.ndim == 4:
        images = rearrange(images, "b c h w -> b h w c")
    elif images.ndim == 5:
        images = rearrange(images, "b c t h w -> b t h w c")
    return images
