import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries


def example_figure(path, row, image, segments, phi_point, phi_cr, curves):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    fig.subplots_adjust(top=.73, right=.9, hspace=.25, wspace=.25)
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[1].imshow(mark_boundaries(image, segments))
    axes[1].set_title('Shared SLIC segmentation')
    limit = max(float(np.max(abs(phi_point))), float(np.max(abs(phi_cr))), 1e-10)
    for ax, phi, title in [(axes[2], phi_point, 'Pointwise SHAP'), (axes[3], phi_cr, 'CR-SHAP')]:
        ax.imshow(image)
        plot = ax.imshow(phi[segments], cmap='coolwarm', vmin=-limit, vmax=limit, alpha=.65)
        ax.set_title(title)
    fig.colorbar(plot, cax=fig.add_axes([.92,.25,.012,.3]), label='Raw cosine SHAP contribution')
    for method, color in [('pointwise', 'tab:blue'), ('cr_shap', 'tab:orange')]:
        curve = curves[method]
        axes[4].plot(curve['fractions'], curve['preferences'], color=color, marker='.',
                     label=f"{method}: PDAUC={curve['auc']:.4f}")
        if curve['fractions'][-1] < 1:
            axes[4].plot([curve['fractions'][-1], 1], [curve['preferences'][-1]]*2, color=color, linestyle='--')
    axes[4].axhline(.5, color='gray', linestyle=':')
    axes[4].set(xlabel='Actual fraction of image pixels removed', ylabel='Pairwise positive preference',
                xlim=(0,1), ylim=(0,1), title='Dashed tails: AUC integration only')
    axes[4].legend(fontsize=8)
    axes[5].axis('off')
    axes[5].text(0, .95, '\n'.join([
        f"Category: {row['sugarcrepe_category']}", f"Example: {row['example_id']}",
        f"Original positive cosine: {row['original_positive_cosine']:.6f}",
        f"Original negative cosine: {row['original_negative_cosine']:.6f}",
        f"Original margin: {row['original_margin']:.6f}",
        f"Original preference: {row['original_preference']:.6f}",
        f"Pointwise minus CR PDAUC: {row['pdauc_difference']:+.6f}",
        f"Positive area: point={row['pointwise_positive_area_fraction']:.1%}, CR={row['cr_shap_positive_area_fraction']:.1%}"]),
        va='top', fontsize=10)
    for ax in axes[:4]:
        ax.axis('off')
    title = 'T+: ' + '\n'.join(textwrap.wrap(row['positive_caption'], 125)) + '\nT−: ' + '\n'.join(textwrap.wrap(row['negative_caption'], 125))
    fig.suptitle(title, y=.98, fontsize=12)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def margin_scatter(path, rows, correlation):
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = sorted({r['sugarcrepe_category'] for r in rows})
    for category in categories:
        subset = [r for r in rows if r['sugarcrepe_category'] == category]
        ax.scatter([r['original_margin'] for r in subset], [r['pdauc_difference'] for r in subset],
                   label=category, alpha=.8)
    ax.axhline(0, color='gray', linestyle='--')
    rho, p = correlation['spearman_rho'], correlation['pvalue']
    ax.set(xlabel='Original positive minus negative cosine margin', ylabel='Pointwise minus CR-SHAP PDAUC',
           title=f'Exploratory relationship: Spearman rho={rho:.3f}, p={p:.3g}' if rho is not None else 'Exploratory relationship (constant input)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
