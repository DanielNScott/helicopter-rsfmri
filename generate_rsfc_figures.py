"""Generate revised Figure 5: rsFC data and prediction results (9-panel).

Each panel is saved as an individual SVG, then compiled into a grid.
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from paths import *
from networks import reassemble_correlations, networks_to_inspect, short_names
from plots import plot_network_aggregated_matrix, plot_LOOCV_AUCs
from svgtools import combine_svgs_horizontal, combine_svgs_vertical, add_text_to_svg, scale_svg, svg_to_pdf

FIGURES_DIR = './data/figures/'
PANEL_WIDTH = 460.8
PANEL_HEIGHT = 345.6
SCALE_TEXT = 2.0
plt.rcParams.update({'font.size': 10 * SCALE_TEXT})


def get_network_ordering(power_csv_file, networks_to_inspect, short_names):
    """Sort ROIs by network and return boundaries and labels for plotting."""
    power_data = pd.read_csv(power_csv_file, delimiter=';')
    systems = power_data['System'].values

    ordered_indices = []
    boundaries = []
    labels = []

    for net, abbrev in zip(networks_to_inspect, short_names):
        start = len(ordered_indices)
        net_rois = np.where(systems == net)[0]
        ordered_indices.extend(net_rois.tolist())
        if len(net_rois) > 0:
            mid = start + len(net_rois) // 2
            labels.append((mid, abbrev))
            boundaries.append(start)

    remaining = [i for i in range(264) if i not in ordered_indices]
    if remaining:
        start = len(ordered_indices)
        ordered_indices.extend(remaining)
        labels.append((start + len(remaining) // 2, 'Other'))
        boundaries.append(start)

    return np.array(ordered_indices), boundaries[1:], labels


def plot_single_pc_matrix(pc_loadings, network_labels, ax, title=''):
    """Plot one PC's loadings as a network x network matrix."""
    n_nets = len(network_labels)
    mat = np.zeros((n_nets, n_nets))
    idx = 0
    for r in range(n_nets):
        for c in range(r, n_nets):
            mat[r, c] = pc_loadings[idx]
            mat[c, r] = pc_loadings[idx]
            idx += 1

    vmax = np.max(np.abs(mat))
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(n_nets))
    ax.set_xticklabels(network_labels, rotation=45, ha='right')
    ax.set_yticks(range(n_nets))
    ax.set_yticklabels(network_labels)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)


def _save_panel(fig, label):
    """Save a figure as an SVG panel."""
    import os
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = FIGURES_DIR + f'fig5_{label}.svg'
    fig.tight_layout()
    fig.savefig(path, format='svg')
    plt.close(fig)
    print(f"  Saved panel {label}")
    return path


def generate_panels():
    """Generate all 9 panels as individual SVG files."""

    # Load data
    data = pd.read_csv("../data/subjX_withid.csv")
    n_subj = data.shape[0]
    correlation_matrices, _, _ = reassemble_correlations(data, n_subj)
    order, boundaries, labels = get_network_ordering(power_csv_file, networks_to_inspect, short_names)
    sorted_matrices = correlation_matrices[:, order, :][:, :, order]
    net_corrs = pd.read_csv("../data/functional_network_corrs.csv")
    with open("./data/cvpca.pickle", "rb") as f:
        cvpca = pickle.load(f)

    roc_data = []
    roc_files = ['data/fig5_raw_pca.pkl', 'data/fig5_network_agg.pkl', 'data/fig5_consensus_pca.pkl']
    roc_titles = ['ROI Correlation PCs', 'Network Aggregates', 'Consensus PCA']
    for fpath in roc_files:
        with open(fpath, 'rb') as f:
            roc_data.append(pickle.load(f))

    # (A) ROI x ROI mean correlation matrix
    fig, ax = plt.subplots()
    mean_corr = np.mean(sorted_matrices, axis=0)
    im = ax.imshow(mean_corr, cmap='RdBu_r', vmin=-0.3, vmax=0.7, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8)
    if labels:
        positions = [pos for pos, _ in labels]
        names = [name for _, name in labels]
        ax.set_xticks([])
        ax.set_yticks(positions)
        ax.set_yticklabels(names)
    if boundaries:
        for b in boundaries:
            ax.axhline(b - 0.5, color='k', linewidth=0.5, alpha=0.5)
            ax.axvline(b - 0.5, color='k', linewidth=0.5, alpha=0.5)
    ax.set_title('ROI x ROI Mean Connectivity')
    _save_panel(fig, 'A')

    # (B) Network-aggregated mean correlation matrix
    fig = plot_network_aggregated_matrix(net_corrs.values, network_labels=short_names)
    _save_panel(fig, 'B')

    # (C) Scree plot with bootstrap PC correlations on right axis
    fig, ax = plt.subplots()
    consensus_var = cvpca['consensus_ve']
    var_ci_lower = cvpca['variance_ci_lower']
    var_ci_upper = cvpca['variance_ci_upper']
    n_components = len(consensus_var)
    x = np.arange(n_components)
    yerr_lower = consensus_var - var_ci_lower
    yerr_upper = var_ci_upper - consensus_var
    ax.errorbar(x, consensus_var, yerr=[yerr_lower, yerr_upper], fmt='o-')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    ax.set_title('Variance Explained')
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    corder = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax2.errorbar(x, cvpca['loading_corrs_avg'], yerr=2 * cvpca['loading_corrs_sem'],
                 fmt='o-', color=corder[1])
    ax2.set_ylabel('Bootstrap PC Correlation')
    _save_panel(fig, 'C')

    # (D, E, F) PC1, PC2, PC3 as network x network matrices
    ve = cvpca['consensus_ve']
    for i, panel_label in enumerate(['D', 'E', 'F']):
        fig, ax = plt.subplots()
        ve_pct = ve[i] * 100
        plot_single_pc_matrix(cvpca['consensus_pcs'][i], short_names, ax, title=f'PC {i+1} ({ve_pct:.0f}%)')
        _save_panel(fig, panel_label)

    # (G, H, I) ROC curves
    corder = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Staircase layout: first outcome top-right, descending left
    # Stats in upper-left, stacked vertically top-down
    text_positions = [(0.02, 0.95), (0.02, 0.865), (0.02, 0.78)]
    stats_fontsize = 10 * SCALE_TEXT * 0.5 + 2
    panel_labels = ['G', 'H', 'I']
    for i, panel_label in enumerate(panel_labels):
        fig, ax = plt.subplots()
        plt.sca(ax)
        results = roc_data[i]
        fit_results_loocv = results['fit_results']['loocv']
        outcomes_list = results['outcomes']

        for j, var in enumerate(outcomes_list):
            plot_LOOCV_AUCs(
                fit_results_loocv[var]['test_preds'],
                fit_results_loocv[var]['test_targs'],
                newfig=False, text=False,
                label=var
            )
            # Place stats text in staircase pattern
            fr = fit_results_loocv[var]
            predictions = np.concatenate(fr['test_preds'])
            ground_truth = np.concatenate(fr['test_targs'])
            from sklearn.metrics import roc_curve, auc
            import scipy as sp
            fpr, tpr, _ = roc_curve(ground_truth, predictions)
            pred_auc = auc(fpr, tpr)
            rpb = sp.stats.pointbiserialr(ground_truth, predictions)
            tx, ty = text_positions[j]
            ax.text(tx, ty, f'(AUC, Rpb, p) = ({pred_auc:.2f}, {rpb[0]:.2f}, {rpb[1]:.2f})',
                    color=corder[j], transform=ax.transAxes, va='top',
                    fontsize=stats_fontsize)
        ax.set_title(roc_titles[i])
        ax.legend(loc='lower right')
        _save_panel(fig, panel_label)

    print("All panels saved.")


def _combine_row(panels, output):
    """Combine panel files horizontally."""
    if len(panels) == 1:
        import shutil
        shutil.copy(panels[0], output)
        return

    tmp = output.replace('.svg', '_tmp.svg')
    combine_svgs_horizontal(panels[0], panels[1], tmp if len(panels) > 2 else output)

    for i, panel in enumerate(panels[2:], start=2):
        src = tmp
        dst = output if i == len(panels) - 1 else tmp.replace('_tmp', f'_tmp{i}')
        combine_svgs_horizontal(src, panel, dst)
        import os
        if os.path.exists(src) and src != dst:
            os.remove(src)
        tmp = dst


def _combine_grid(rows, output):
    """Combine row files vertically."""
    if len(rows) == 1:
        import shutil
        shutil.copy(rows[0], output)
        return

    tmp = output.replace('.svg', '_tmp.svg')
    combine_svgs_vertical(rows[0], rows[1], tmp if len(rows) > 2 else output)

    for i, row in enumerate(rows[2:], start=2):
        src = tmp
        dst = output if i == len(rows) - 1 else tmp.replace('_tmp', f'_tmp{i}')
        combine_svgs_vertical(src, row, dst)
        import os
        if os.path.exists(src) and src != dst:
            os.remove(src)
        tmp = dst


def compile_figure():
    """Compile individual panels into a 3x3 grid figure."""
    import os

    layout = [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']]
    labels = [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']]

    # Build rows
    row_files = []
    for r, row in enumerate(layout):
        row_panels = [FIGURES_DIR + f'fig5_{label}.svg' for label in row]
        row_file = FIGURES_DIR + f'fig5_row{r}.svg'
        _combine_row(row_panels, row_file)
        row_files.append(row_file)

    # Combine rows vertically
    combined = FIGURES_DIR + 'fig5_combined.svg'
    _combine_grid(row_files, combined)

    # Add panel labels
    labeled = FIGURES_DIR + 'fig5_labeled.svg'
    current = combined
    for r, row in enumerate(labels):
        y = sum([PANEL_HEIGHT] * r) + 20
        for c, label in enumerate(row):
            if label is None:
                continue
            x = sum([PANEL_WIDTH] * c) + 10
            add_text_to_svg(current, labeled, label, x=x, y=y, font_size=21)
            current = labeled

    # Scale to publication width and convert to PDF
    final = FIGURES_DIR + 'fig5_final.svg'
    scale_svg(labeled, final, FIG_WIDTH=6.27)
    svg_to_pdf(final, final.replace('.svg', '.pdf'))

    # Clean up intermediates
    for f in row_files + [combined, labeled]:
        if os.path.exists(f):
            os.remove(f)

    print(f"Saved {final} and {final.replace('.svg', '.pdf')}")


def main():
    generate_panels()
    compile_figure()


if __name__ == "__main__":
    main()
