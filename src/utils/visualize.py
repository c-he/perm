from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from matplotlib.colors import Normalize

# plt.rcParams.update({
#     "text.usetex": True,  # require LaTeX and type1cm on Ubuntu
#     "font.family": "sans-serif",
#     'text.latex.preamble': [
#         r"""
#         \usepackage{libertine}
#         \usepackage[libertine]{newtxmath}
#         """],
# })


def config_plot():
    """ Function to remove axis tickers and box around figure.
    """
    plt.box(False)
    plt.axis('off')
    plt.tight_layout()


def plot_point_clouds(fp: str, points: np.ndarray, seed_points: Optional[np.ndarray] = None, colors: Optional[np.ndarray] = None) -> None:
    """ Visualize points with their labels as different colors.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(points[:, 0], points[:, 1], s=5, c=colors if colors is not None else 'aqua')
    if seed_points is not None:
        ax.scatter(seed_points[:, 0], seed_points[:, 1], s=10, c='black')
    ax.invert_yaxis()
    config_plot()
    plt.savefig(fp, dpi=300)
    plt.close()


def plot_explained_variance(fp: str, expalined_variance_ratio: np.ndarray) -> None:
    cumulative_variance = expalined_variance_ratio.cumsum()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(1, cumulative_variance.shape[0] + 1), cumulative_variance, linewidth=1)
    ax.set_ylim(top=1.0)
    ax.margins(x=0, y=0)
    ax.grid(linestyle='--')

    ax.set_title('Cumulative Relative Variance')
    ax.set_xlabel('\# of Principal Components')
    ax.set_ylabel('Cumulative Variance Ratio')

    plt.tight_layout()
    plt.savefig(fp, dpi=300)
    plt.close()


def plot_weight_std(fp: str, weight_image: torch.Tensor, colorbar=False) -> None:
    """ Visualize the standard deviation of blending weights in the shape [C, H, W].
    """
    std = weight_image.std(dim=0)
    cmap = cm.jet
    norm = Normalize(vmin=0, vmax=1)
    std = cmap(norm(std.cpu().numpy()))[..., :3]

    if colorbar:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(std)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), fraction=0.0455, pad=0.04, location='left')
        config_plot()
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        std = np.rint(std * 255).clip(0, 255).astype(np.uint8)
        PIL.Image.fromarray(std, 'RGB').save(fp)


def plot_texture_error(fp: str, error_image: torch.Tensor, colorbar=False) -> None:
    """ Visualize the error image in the shape [H, W].
    """
    cmap = cm.jet
    # norm = Normalize(vmin=error_image.min(), vmax=error_image.max())
    norm = Normalize(vmin=0, vmax=150)
    image = cmap(norm(error_image.cpu().numpy()))[..., :3]

    if colorbar:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(image)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), fraction=0.0455, pad=0.04, location='left')
        config_plot()
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        image = np.rint(image * 255).clip(0, 255).astype(np.uint8)
        PIL.Image.fromarray(image, 'RGB').save(fp)
