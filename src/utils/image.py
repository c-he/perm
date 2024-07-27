import glob
import os

import numpy as np
import PIL.Image
import torch

""" A module for reading / writing various image formats. """


def write_exr(path, image):
    """ Write an EXR image to some path.
        Image is a dict of form { "default" = rgb_array, "depth" = depth_array }

    Args:
        path (str): Path to save the EXR
        image (dict): Dictionary of EXR buffers.

    Returns:
        (void): Write to path.
    """
    try:
        import pyexr
    except:
        raise Exception(
            "Module pyexr is not available. To install, run `pip install pyexr`. "
            "You will likely also need `libopenexr`, which through apt you can install with "
            "`apt-get install libopenexr-dev` and on Windows you can install with "
            "`pipwin install openexr`.")
    pyexr.write(path, image,
                channel_names={'normal': ['X', 'Y', 'Z'],
                               'x': ['X', 'Y', 'Z'],
                               'view': ['X', 'Y', 'Z']},
                precision=pyexr.HALF)


def write_texture(path, texture, alpha=None):
    """ Write the first 3 channels of a multi-channel texture as a PNG image.
    Args:
        texture (torch.Tensor): HWC data tensors.
        alpha (torch.Tensor): alpha mattes.
    """
    lo, high = texture.min(), texture.max()
    texture = (texture - lo) / (high - lo)
    texture = texture[..., :3]
    if alpha is not None:
        texture = torch.cat([texture, alpha], dim=-1)

    write_png(path, texture)


def write_png(path, image):
    """ Write a PNG image to some path.

    Args:
        path (str): Path to save the PNG.
        image (torch.Tensor): HWC image.

    Returns:
        (void): Write to path.
    """
    image = (image * 255.0).clamp(0, 255).to(torch.uint8)
    C = image.shape[-1]
    if C == 1:
        PIL.Image.fromarray(image[..., 0].cpu().numpy(), 'L').save(path)
    elif C == 4:
        PIL.Image.fromarray(image.cpu().numpy(), 'RGBA').save(path)
    else:
        PIL.Image.fromarray(image.cpu().numpy(), 'RGB').save(path)


def glob_imgs(path, exts=['*.png', '*.PNG', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG']):
    """ Utility to find images in some path.

    Args:
        path (str): Path to search images in.
        exts (list of str): List of extensions to try.

    Returns:
        (list of str): List of paths that were found.
    """
    imgs = []
    for ext in exts:
        imgs.extend(glob.glob(os.path.join(path, ext)))
    return imgs


def load_rgb(path, normalize=True):
    """ Load an image.

    Args:
        path (str): Path to the image.
        noramlize (bool): If True, will return [0,1] floating point values. Otherwise returns [0,255] ints.

    Returns:
        (np.array): Image as an array of shape [H,W,C]
    """
    image = np.array(PIL.Image.open(path))
    if normalize:
        image = image.astype(np.float32) / 255.0
    return image


def hwc_to_chw(img):
    """ Convert [H,W,C] to [C,H,W] for TensorBoard output.

    Args:
        img (torch.Tensor): [H,W,C] image.

    Returns:
        (torch.Tensor): [C,H,W] image.
    """
    return img.permute(2, 0, 1)


def chw_to_hwc(img):
    """ Convert [C,H,W] to [H,W,C].

    Args:
        img (torch.Tensor): [C,H,W] image.

    Returns:
        (torch.Tensor): [H,W,C] image.
    """
    return img.permute(1, 2, 0)
