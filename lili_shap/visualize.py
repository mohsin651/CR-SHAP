import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries

from .core import METHODS


def explanation_figure(path, image, segments, values, curves, caption):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[1].imshow(mark_boundaries(image, segments))
    axes[1].set_title('Shared SLIC segmentation')
    limit = max(max(abs(x)) for x in values.values()) or 1
    for ax, method in zip(axes[2:5], METHODS):
        ax.imshow(image)
        plot = ax.imshow(values[method][segments], cmap='coolwarm', vmin=-limit, vmax=limit, alpha=0.65)
        ax.set_title(method + ' SHAP')
    fig.subplots_adjust(right=0.89, hspace=0.3, wspace=0.25)
    color_axis = fig.add_axes([0.92, 0.35, 0.012, 0.3])
    fig.colorbar(plot, cax=color_axis, label='SHAP cosine contribution')
    for method in METHODS:
        curve = curves[method]
        axes[5].plot(curve['auc_fractions'], curve['auc_scores'], marker='.',
                     label=f"{method}: AUC={curve['auc']:.4f}")
    axes[5].set(xlabel='Fraction of image pixels removed', ylabel='CLIP cosine similarity')
    axes[5].legend(fontsize=8)
    for ax in axes[:5]:
        ax.axis('off')
    fig.suptitle(caption, wrap=True)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def perturbation_figure(path, image, mask, expanded, examples):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, title, data in zip(axes.ravel(),
        ['Original', 'Absent-superpixel mask', 'Expanded mask', *METHODS],
        [image, mask, expanded, *[examples[m] for m in METHODS]]):
        ax.imshow(data, cmap='gray' if data.ndim == 2 else None, vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
