import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_mean_correlation_matrix(correlation_matrices, network_labels=None, network_boundaries=None, figsize=(8, 7)):
    """Plot across-participant mean ROI x ROI correlation matrix.

    correlation_matrices: (n_subj, 264, 264) array
    network_labels: list of (position, name) tuples for network tick marks
    network_boundaries: list of ROI indices where networks change
    """
    mean_corr = np.mean(correlation_matrices, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(mean_corr, cmap='RdBu_r', vmin=-0.3, vmax=0.7, aspect='auto')
    plt.colorbar(im, ax=ax, label='Mean Fisher Z correlation', shrink=0.8)

    # Draw network boundaries
    if network_boundaries is not None:
        for b in network_boundaries:
            ax.axhline(b - 0.5, color='k', linewidth=0.5, alpha=0.5)
            ax.axvline(b - 0.5, color='k', linewidth=0.5, alpha=0.5)

    # Label networks
    if network_labels is not None:
        positions = [pos for pos, _ in network_labels]
        names = [name for _, name in network_labels]
        ax.set_xticks(positions)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yticks(positions)
        ax.set_yticklabels(names)
    else:
        ax.set_xlabel('ROI')
        ax.set_ylabel('ROI')

    ax.set_title('Mean ROI x ROI Functional Connectivity')
    plt.tight_layout()
    return fig


def plot_pca_components_matrix(consensus_pcs, network_labels=None, n_components=9, figsize=(10, 8)):
    """Plot PCA component loadings reshaped as network x network matrices.

    consensus_pcs: (n_components, 28) array of PC loadings over network pairs
    network_labels: list of 7 short network names
    """
    if network_labels is None:
        network_labels = ['DMN', 'CO', 'VA', 'Vis', 'FPN', 'Sal', 'DAN']

    n_nets = len(network_labels)
    n_cols = min(3, n_components)
    n_rows = int(np.ceil(n_components / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for i in range(n_components):
        ax = axes[i // n_cols, i % n_cols]

        # Reconstruct symmetric matrix from upper triangle
        mat = np.zeros((n_nets, n_nets))
        idx = 0
        for r in range(n_nets):
            for c in range(r, n_nets):
                mat[r, c] = consensus_pcs[i, idx]
                mat[c, r] = consensus_pcs[i, idx]
                idx += 1

        vmax = np.max(np.abs(mat))
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(range(n_nets))
        ax.set_xticklabels(network_labels, rotation=45, ha='right')
        ax.set_yticks(range(n_nets))
        ax.set_yticklabels(network_labels)
        ax.set_title(f'PC {i + 1}')
        plt.colorbar(im, ax=ax, shrink=0.7)

    # Hide unused axes
    for i in range(n_components, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.suptitle('Consensus PCA Component Loadings (Network x Network)')
    plt.tight_layout()
    return fig


def plot_network_aggregated_matrix(flat_corrs, network_labels=None, ax=None, figsize=None):
    """Plot mean network-aggregated correlation matrix (7x7).

    flat_corrs: (n_subj, 28) array of network-pair correlations
    network_labels: list of 7 short network names
    """
    if network_labels is None:
        network_labels = ['DMN', 'CO', 'VA', 'Vis', 'FPN', 'Sal', 'DAN']

    n_nets = len(network_labels)
    mean_vals = np.mean(flat_corrs, axis=0)

    # Reconstruct symmetric matrix from upper triangle
    mat = np.zeros((n_nets, n_nets))
    idx = 0
    for r in range(n_nets):
        for c in range(r, n_nets):
            mat[r, c] = mean_vals[idx]
            mat[c, r] = mean_vals[idx]
            idx += 1

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(mat, cmap='RdBu_r', vmin=-0.05, vmax=0.2, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n_nets))
    ax.set_xticklabels(network_labels, rotation=45, ha='right')
    ax.set_yticks(range(n_nets))
    ax.set_yticklabels(network_labels)
    ax.set_title('Mean Network Connectivity')
    return fig


def plot_pc_loadings_points(consensus_pcs, consensus_ve, ci_lower, ci_upper, feature_names, n_components=4, ax=None, figsize=(6, 4)):
    """Plot PC loadings as points with error bars, dim if CI overlaps zero.

    consensus_pcs: (n_pcs, 28) array
    consensus_ve: (n_pcs,) variance explained
    ci_lower, ci_upper: (n_pcs, 28) arrays of 95% CI bounds
    feature_names: list of 28 network-pair names
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    n_features = len(feature_names)
    x = np.arange(n_features)
    spacing = 0.8 / n_components

    for i in range(n_components):
        offset = (i - n_components / 2 + 0.5) * spacing
        ve_pct = consensus_ve[i] * 100
        y = consensus_pcs[i]
        lo = ci_lower[i]
        hi = ci_upper[i]
        yerr_lo = y - lo
        yerr_hi = hi - y

        # Determine which loadings have CIs overlapping zero
        overlaps_zero = (lo <= 0) & (hi >= 0)

        # Plot non-overlapping points at full opacity
        color = f'C{i}'
        mask = ~overlaps_zero
        if np.any(mask):
            ax.errorbar(
                x[mask] + offset, y[mask],
                yerr=[yerr_lo[mask], yerr_hi[mask]],
                fmt='o', color=color, markersize=4, capsize=2,
                label=f'PC{i+1} ({ve_pct:.0f}%)', alpha=1.0,
            )

        # Plot zero-overlapping points dimmed
        mask = overlaps_zero
        if np.any(mask):
            ax.errorbar(
                x[mask] + offset, y[mask],
                yerr=[yerr_lo[mask], yerr_hi[mask]],
                fmt='o', color=color, markersize=4, capsize=2,
                alpha=0.3,
            )

    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=90)
    ax.set_ylabel('Loading')
    ax.set_title('Consensus PC Loadings')
    ax.legend(loc='upper right')
    return fig


def plot_iolike_by_forgetfulness(data):
    plt.plot(data['forgetfulness'], data['IOLike']+np.random.randn(265)*0.05,'o')
    plt.xlabel('Forgetfulness')
    plt.ylabel()
    plt.ylabel('IOLike, Jittered')
    plt.title('Forgetfulness by IOLike')


def plot_LOOCV_AUCs(test_preds, test_outcomes, newfig=True, text=True, text_offset=0.0, text_color='k', label=''):
    # For LOOCV, compute AUC on full prediction list
    predictions  = np.concatenate(test_preds)
    ground_truth = np.concatenate(test_outcomes)

    fpr, tpr, _ = roc_curve(ground_truth, predictions)
    pred_auc = auc(fpr, tpr)

    rpb = sp.stats.pointbiserialr(ground_truth, predictions)

    if newfig: plt.figure()
    plt.plot([0,1], [0,1], 'k--')
    plt.plot(fpr, tpr, label=label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LOO Prediction ROC Curve')
    if text: plt.text(0.7, 0.0+text_offset, f'AUC = {pred_auc:.3f}\nRpb = {rpb[0]:.3f}\np = {rpb[1]:.3f}', color=text_color)

def plot_LOOCV_PR_curves(test_preds, test_outcomes, newfig=True, text=True, text_offset=0.0, text_color='k', label=''):
    # For LOOCV, compute PR curve on full prediction list
    predictions  = np.concatenate(test_preds)
    ground_truth = np.concatenate(test_outcomes)
    precision, recall, _ = precision_recall_curve(ground_truth, predictions)
    pred_auc = auc(recall, precision)
    rpb = sp.stats.pointbiserialr(ground_truth, predictions)
    
    # Baseline for PR curve is the prevalence (proportion of positive class)
    prevalence = np.mean(ground_truth)

    # Fully normalized AUC over prevalence
    pred_auc = (pred_auc - prevalence)/(1-prevalence)

    recall = np.flip(recall)
    precision = np.flip(precision)

    ind = (len(precision)//20)

    if newfig: plt.figure()
    plt.plot(recall[ind:], (precision[ind:]-prevalence)*100/(1-prevalence), label=label)
    plt.xlabel('Recall (True Positive Rate)')
    plt.ylabel('Normed Improvement over Base Rate') 
    plt.title('LOO Prediction PR Curve')
    if text: plt.text(0.6, 20+text_offset, f'NAUC_PR = {pred_auc:.2f}', color=text_color)


def plot_analysis_results(fit_results, auc_mean, auc_sem, outcomes):
    # Get default line plotting color order
    corder = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Open a figure for LOOCV PRs and AUCs, plus 10-fold CV AUCs
    plt.figure(figsize=(12,4))

    if 'loocv' in fit_results.keys():
        # Plot PR curves for each variable
        plt.subplot(1,3,1)
        text_offsets, text_colors = [5,0,-5], [corder[0], corder[1], corder[2]]
        for i, var in enumerate(outcomes):
            # Extract data for this variable
            y_test = fit_results['loocv'][var]["test_targs"]
            y_pred = fit_results['loocv'][var]["test_preds"]

            # Plot this variable's data
            plot_LOOCV_PR_curves(y_pred, y_test, text=True, text_offset=text_offsets[i], text_color=text_colors[i], newfig=False, label=var)
        
        plt.axhline(0, linestyle='--', color='k', alpha=0.5, label='Prevalence')
        plt.legend()

        plt.subplot(1,3,2)
        text_offsets = [0.4, 0.2, 0.0]
        for i, var in enumerate(outcomes):
            y_test = fit_results['loocv'][var]["test_targs"]
            y_pred = fit_results['loocv'][var]["test_preds"]
            plot_LOOCV_AUCs(y_pred, y_test, text=True, text_offset=text_offsets[i], text_color=text_colors[i], newfig=False, label=var)
        plt.legend()

    plt.subplot(1,3,3)
    plt.bar(outcomes, [auc_mean[var] for var in outcomes], yerr=[2*auc_sem[var] for var in outcomes])
    plt.ylabel('AUC')
    plt.title('Means of 10-Fold Held-Out AUCs')
    plt.legend(['AUC +- 2 SEM'])
    plt.axhline(0.5,linestyle='--',color='r',alpha=0.5,zorder=10)

    plt.tight_layout()



def plot_loocvs_from_results_only(fit_results, outcomes):
    # Get default line plotting color order

    corder = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Open a figure for LOOCV PRs and AUCs, plus 10-fold CV AUCs
    plt.figure(figsize=(4,4))

    if 'loocv' in fit_results.keys():
        text_offsets = [0.4, 0.2, 0.0]
        text_colors = [corder[0], corder[1], corder[2]]
        for i, var in enumerate(outcomes):
            y_test = fit_results['loocv'][var]["test_targs"]
            y_pred = fit_results['loocv'][var]["test_preds"]
            plot_LOOCV_AUCs(y_pred, y_test, text=True, text_offset=text_offsets[i], text_color=text_colors[i], newfig=False, label=var)
        plt.legend()
        plt.tight_layout()


def plot_loocv_regression(fit_results, outcomes):
    """Plot LOOCV predicted vs actual for continuous outcomes."""
    from analysis import get_loocv_regression_stats

    plt.figure(figsize=(4 * len(outcomes), 4))
    for i, var in enumerate(outcomes):
        test_preds = np.concatenate(fit_results['loocv'][var]['test_preds'])
        test_targs = np.concatenate(fit_results['loocv'][var]['test_targs'])
        stats = get_loocv_regression_stats(fit_results['loocv'][var])

        plt.subplot(1, len(outcomes), i + 1)
        plt.scatter(test_targs, test_preds, alpha=0.3, s=10)
        xlim = plt.xlim()
        plt.plot(xlim, xlim, 'k--', alpha=0.5)
        plt.xlim(xlim)
        plt.xlabel('Actual')
        plt.ylabel('Predicted (LOO)')
        plt.title(var)
        plt.text(0.05, 0.95, f"r = {stats['r']:.3f}\nR² = {stats['r_squared']:.3f}\np = {stats['p']:.3f}",
            transform=plt.gca().transAxes, va='top')
        plt.grid(alpha=0.3)

    plt.tight_layout()


def plot_regression_stability(outcomes, use_selections, selections, flatcorrcols, coeffs_avg, coeffs_std):
    # Plot the regression model stability over folds
    plt.figure(figsize=(12,4))
    for i, var in enumerate(outcomes):
        if use_selections:
            colnames = ['const'] + [pred for pred in selections[var].index if selections[var][pred]]
        else:
            colnames = ['const'] + flatcorrcols
        plt.subplot(1,3,i+1)
        plt.errorbar(colnames, y=coeffs_avg[var], yerr=2*coeffs_std[var], fmt='o')
        plt.xticks(rotation=45, ha='right')
        plt.grid(alpha=0.3)
        plt.legend(['Mean +- 2*SD'])
        plt.title('IOLike Model Stability')
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.ylabel("Standard Beta")
        plt.xlabel("Variable")
        plt.tight_layout()



def plot_selection_analysis(selection_counts, aic_means, aic_stds, n_folds, show_sems=True):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Get outcomes and predictors from DataFrame structure
    outcomes = selection_counts.columns.tolist()
    predictors = selection_counts.index.tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(outcomes)))
    x_positions = np.arange(len(predictors))
    
    # Plot 1: Selection counts
    for i, outcome in enumerate(outcomes):
        counts = selection_counts[outcome].values  # Get all counts for this outcome
        offset = (i - len(outcomes)/2 + 0.5) * 0.15
        ax1.plot(x_positions + offset, counts, 'o', color=colors[i], 
                 label=outcome, markersize=6)
    
    ax1.set_xlabel('Variables')
    ax1.set_ylabel('Selection Count')
    ax1.set_title('Variable Selection Frequency Across Folds')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(predictors, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: AIC impacts
    for i, outcome in enumerate(outcomes):
        
        # Get the y-data (AIC impacts)
        means = aic_means[outcome].values
        sds   = aic_stds[outcome].values
        sems  = sds/np.sqrt(n_folds)
        
        # Compute x-offsets
        offset = (i - len(outcomes)/2 + 0.5) * 0.2
        x_pos_offset = x_positions + offset
        
        # Plot AIC impacts for this outcome variable
        ax2.errorbar(x_pos_offset, means, yerr=2*sds, fmt='o', color=colors[i], label=outcome + ' Mean +- 2SD', markersize=6)
        #ax2.errorbar(x_pos_offset, means, yerr=2*sems, fmt='.', color='m', markersize=0.1, capsize=2)

    ax2.set_xlabel('Variables')
    ax2.set_ylabel('AIC Impact (Mean ± 2SD)')
    ax2.set_title('Variable AIC Impact')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(predictors, rotation=45, ha='right')
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



def plot_regression_predictions(X, data, preds_avg, preds_std, preds_loo):
    from analysis import fit_logistic_regression
    n_subj = X.shape[0]

    for var in ['IOLike', 'RWLike', 'CPLike']:

        y = data[var]
        _, _, y_hat = fit_logistic_regression(X, y)

        fpr, tpr, _ = roc_curve(y, y_hat)
        aucs = auc(fpr, tpr)

        # Sort subjects on their predicted trait value
        inds = np.argsort(preds_avg[var])
        sids = np.arange(n_subj)

        plt.figure(figsize=(12,8))

        plt.subplot(2,3,1)
        plt.plot(sids, preds_avg[var][inds], label='Average')
        plt.fill_between(sids, preds_avg[var][inds]-2*preds_std[var][inds], preds_avg[var][inds]+2*preds_std[var][inds], alpha=0.3, label='+- 2 SD')
        plt.scatter(sids, preds_loo[var][inds], color='red', s=1, label='LOO Prediction')
        plt.xlabel("Sorted Subject Index")
        plt.ylabel("IOLike Prediction")
        plt.title("IOLike Training vs Testing")

        plt.subplot(2,3,4)
        plt.plot(fpr, tpr, label=f'AUC = {aucs:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Full Model')
        plt.legend()
        plt.show()

        plt.subplot(2,3,5)
        corrcoef = np.corrcoef(y_hat, preds_avg[var])[0,1]
        plt.scatter(y_hat, preds_avg[var], alpha=0.3, s=5)
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('Predictions, Full Model')
        plt.ylabel('Avg. Predictions, LOO Training')
        plt.title('LOO Impact on Training Predictions')

        plt.subplot(2,3,6)
        corrcoef = np.corrcoef(preds_avg[var], preds_loo[var])[0,1]
        plt.scatter(preds_avg[var], preds_loo[var], alpha=0.3, s=5)
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('Avg. Predictions, LOO Training')
        plt.ylabel('Predictions, LOO Test')
        plt.title('LOO Training-Testing Discrepancy')

        plt.tight_layout()

