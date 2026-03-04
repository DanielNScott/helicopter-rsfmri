import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.linalg import orthogonal_procrustes
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde
import matplotlib. pyplot as plt
import pickle

from networks import *
from paths import *
from dataio import load_and_merge_raw_data

from networks import get_functional_networks

def fit_PCA(X):
    # Set up and fit the PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X_scaled)

    # Get PCA results
    pcs    = pca.components_
    ve     = pca.explained_variance_ratio_

    # Reorder PCs by variance explained
    order = np.argsort(-ve)
    pcs = pcs[order]
    ve  = ve[order]

    # Get subject scores
    scores = pca.transform(X_scaled)

    return pcs, ve, scores


def bootstrap_consensus_pca(X, n_bootstrap=100, alignment='correlation', random_state=None):
    """
    Perform bootstrap consensus PCA to get stable comps with confidence intervals.
    Returns consensus loadings and variance explained with bootstrap confidence intervals.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Data dimensions
    n_samples, n_features = X.shape
    
    # Fit ref PCA on full data
    ref_pcs, ref_ve, ref_scores = fit_PCA(X)

    # Keep track of scores for each participant
    scores = [[] for _ in range(n_samples)]

    # Storage for bootstrap results
    all_sample_pcs = np.zeros((n_bootstrap, n_features, n_features))
    all_sample_ve  = np.zeros((n_bootstrap, n_features))
    
    # Bootstrap loop
    for b in range(n_bootstrap):
        # Generate bootstrap sample
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X[bootstrap_indices]
        
        # Fit PCA on bootstrap sample
        sample_pcs, sample_ve, _ = fit_PCA(X_bootstrap)

        # Align comps to ref
        aligned_pcs, aligned_ve = align_comps_correlation(sample_pcs, sample_ve, ref_pcs)

        
        all_sample_pcs[b] = aligned_pcs
        all_sample_ve[b]  = aligned_ve

        # Compute participant scores on aligned PCs
        participant_scores = np.dot(X_bootstrap, aligned_pcs.T)

        # Copy participant scores, avoiding multiple copies from the same bootstrap
        seen = []
        for ind_cur, ind_ref in enumerate(bootstrap_indices):
            if ind_ref not in seen:
                scores[ind_ref].append(participant_scores[ind_cur])
                seen.append(ind_ref)

    # Stack scores
    for i, score_list in enumerate(scores):
        if len(score_list) > 1:
            scores[i] = np.stack(score_list)
        elif len(score_list) == 1:
            scores[i] = np.reshape(score_list, (1, -1))
        else:
            scores[i] = np.full((1, n_features), np.nan)

    # Compute consensus comps and confidence intervals
    consensus_pcs    = np.mean(all_sample_pcs, axis=0)
    consensus_ve     = np.mean(all_sample_ve, axis=0)
    consensus_scores = np.array([np.nanmean(s, axis=0) for s in scores])
    consensus_scores_sds = np.array([np.nanstd(s, axis=0) for s in scores])
    consensus_scores_sem = np.array([np.nanstd(s, axis=0)/np.sqrt(s.shape[0]) for s in scores])

    # Bootstrap confidence intervals (2.5th and 97.5th percentiles)
    component_ci_lower = np.percentile(all_sample_pcs, 2.5, axis=0)
    component_ci_upper = np.percentile(all_sample_pcs, 97.5, axis=0)
    variance_ci_lower  = np.percentile(all_sample_ve, 2.5, axis=0)
    variance_ci_upper  = np.percentile(all_sample_ve, 97.5, axis=0)
    scores_ci_lower    = np.percentile(consensus_scores, 2.5, axis=0)
    scores_ci_upper    = np.percentile(consensus_scores, 97.5, axis=0)

    # Standard deviations (we'll compute SNRs)
    loading_sds = np.std(all_sample_pcs, axis=0)

    # Compute correlations between the consensus PCs and the sample PCs
    loading_corrs = np.array([[np.corrcoef(consensus_pcs[i], all_sample_pcs[j][i])[0,1] for j in range(n_bootstrap)] for i in range(n_features)])
    loading_corrs_avg = np.mean(loading_corrs,axis=1)
    loading_corrs_std = np.std(loading_corrs,axis=1)
    loading_corrs_sem = np.std(loading_corrs,axis=1)/np.sqrt(n_bootstrap)


    results = {
        'consensus_pcs': consensus_pcs,
        'consensus_ve': consensus_ve,
        'consensus_scores': consensus_scores,
        'consensus_scores_sds': consensus_scores_sds,
        'consensus_scores_sem': consensus_scores_sem,
        'scores_ci_lower': scores_ci_lower,
        'scores_ci_upper': scores_ci_upper,
        'loading_sds': loading_sds,
        'component_ci_lower': component_ci_lower,
        'component_ci_upper': component_ci_upper,
        'variance_ci_lower': variance_ci_lower,
        'variance_ci_upper': variance_ci_upper,
        'ref_pcs': ref_pcs,
        'ref_ve': ref_ve,
        'loading_corrs_avg': loading_corrs_avg,
        'loading_corrs_std': loading_corrs_std,
        'loading_corrs_sem': loading_corrs_sem
    }
    
    return results


def align_comps_correlation(sample_pcs, sample_ve, ref_pcs):
    """Align bootstrap comps to ref using correlation matching."""
    n_comps = sample_pcs.shape[0]

    # Initialize new component vectors and variance explained (shares idx with ref)
    aligned_pcs = np.zeros_like(sample_pcs)
    aligned_ve   = np.zeros_like(sample_ve)

    # Get correlation matrix relating bootstrap and ref comps
    correlations = np.corrcoef(sample_pcs, ref_pcs)[:n_comps, n_comps:]
    
    # Assemble a table of index associations
    for _ in range(n_comps):

        # Get the location of the highest correlation
        idx = np.argmax(np.abs(correlations).flatten())
        boot_idx, ref_idx = np.unravel_index(idx, correlations.shape)

        # Assign the component and it's associated variance
        aligned_pcs[ref_idx] = sample_pcs[boot_idx] * np.sign(correlations[boot_idx, ref_idx])
        aligned_ve[ref_idx]  = sample_ve[boot_idx]

        # Set the row and column of the correlation matrix to zero
        correlations[boot_idx, :] = 0
        correlations[:, ref_idx] = 0
    
    return aligned_pcs, aligned_ve


def _align_comps_procrustes(sample_pcs, sample_ve, ref_pcs, ref_ve):
    """Align bootstrap comps to ref using Procrustes analysis."""
    # Procrustes alignment finds optimal rotation matrix
    R, _ = orthogonal_procrustes(sample_pcs.T, ref_pcs.T)
    aligned_pcs = (sample_pcs @ R.T)
    
    # Handle sign flips after Procrustes
    for i in range(aligned_pcs.shape[0]):
        if np.corrcoef(aligned_pcs[i], ref_pcs[i])[0, 1] < 0:
            aligned_pcs[i] *= -1
    
    # Variance stays the same order after Procrustes
    aligned_ve = sample_ve.copy()
    
    return aligned_pcs, aligned_ve

def transform_consensus_pca(X, results):
    """Transform new data using consensus PCA results."""
    X_scaled = results['scaler'].transform(X)
    return X_scaled @ results['consensus_pcs'].T

def plot_pc1_loadings(results, flatcorcols):
    """Plot the loadings of the first principal component."""
    import matplotlib.pyplot as plt
    
    pc1 = results['consensus_pcs'][0]
    ci_lower = results['component_ci_lower'][0]
    ci_upper = results['component_ci_upper'][0]
    
    x = np.arange(len(pc1))
    y = pc1
    yerr_lower = y - ci_lower
    yerr_upper = ci_upper - y
    
    xinds = np.arange(0,28,2)
    #xinds = np.arange(0,28)

    plt.figure(figsize=(4, 4))
    plt.errorbar(x, y, yerr=[yerr_lower, yerr_upper], alpha=0.7, marker='o')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.title('PC 1 Loadings with 95% CI')
    #plt.xlabel('Feature')
    #plt.ylabel('Loading')
    plt.xticks(ticks=xinds, labels=[flatcorcols[i] for i in xinds], rotation=90)
    plt.tight_layout()
    plt.show()


def plot_consensus_loadings(results, component_indices=None, figsize=(12, 8)):
    """Plot consensus component loadings with confidence intervals."""
    import matplotlib.pyplot as plt
    
    consensus_pcs = results['consensus_pcs']
    ci_lower = results['component_ci_lower']
    ci_upper = results['component_ci_upper']
    
    if component_indices is None:
        component_indices = range(min(6, consensus_pcs.shape[0]))
    
    n_plots = len(component_indices)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, comp_idx in enumerate(component_indices):
        ax = axes[i] if n_plots > 1 else axes[0]
        
        x = np.arange(consensus_pcs.shape[1])
        y = consensus_pcs[comp_idx]
        yerr_lower = y - ci_lower[comp_idx]
        yerr_upper = ci_upper[comp_idx] - y
        
        ax.errorbar(x, y, yerr=[yerr_lower, yerr_upper], alpha=0.7, capsize=2, capthick=1)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_title(f'PC {comp_idx + 1}')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Loading')
    
    # Hide extra subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_variance_explained(results, figsize=(10, 5)):
    """Plot variance explained with confidence intervals."""
    import matplotlib.pyplot as plt
    
    consensus_var = results['consensus_ve']
    var_ci_lower = results['variance_ci_lower']
    var_ci_upper = results['variance_ci_upper']

    n_components = np.shape(consensus_var)[0]
    x = np.arange(0, n_components)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Individual variance explained
    yerr_lower = consensus_var - var_ci_lower
    yerr_upper = var_ci_upper - consensus_var

    ax1.plot(x, consensus_var, 'o-', label='Est')
    ax1.fill_between(x, consensus_var-yerr_lower, consensus_var+yerr_upper, alpha=0.3, label='95% CI')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained')
    ax1.set_title('Individual Variance Explained')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Cumulative variance explained
    cumvar = np.cumsum(consensus_var)
    cumvar_lower = np.cumsum(var_ci_lower)
    cumvar_upper = np.cumsum(var_ci_upper)

    ax2.plot(x, cumvar, 'o-', label='Est')
    ax2.fill_between(x, cumvar_lower, cumvar_upper, alpha=0.3, label="95% CI")
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Cumulative Variance Explained')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_pca_reliability(results):
    n_pcs = results["consensus_scores"].shape[1]

    # Compute SNRs for each principal component
    snrs = np.abs(results["consensus_scores"]) / (results["consensus_scores_sds"])
    #
    consensus_pcs = results['consensus_pcs']

    # Calculate stability as average loading d' from zero
    loading_mags = np.abs(consensus_pcs)
    loading_sds  = results['loading_sds']
    signal_to_noise = np.mean(loading_mags / (loading_sds + 1e-10), axis=1)

    plt.figure(figsize=(8, 4))

    # Plot average participant score SNRs
    plt.subplot(1,2,1)
    plt.plot( np.mean(snrs, axis=0), 'o-', label='Score')
    plt.plot(signal_to_noise,  'o-', label="Loading")

    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='SNR = 1')
    plt.xlabel('Principal Component')
    plt.ylabel('Mean SNR')
    plt.title('Score & Loading Reliability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Plot average loading magnitude SNRs
    plt.subplot(1,2,2)
    plt.errorbar(range(n_pcs), results["loading_corrs_avg"], yerr=2*results["loading_corrs_sem"], fmt='o-')
    plt.legend(["Mean +- 2*SEM"])
    plt.xlabel('Principal Component')
    plt.ylabel('Correlation Coefficient')
    plt.title('Bootstrap PC Correlations')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_kdes(snrs):
    # Compute KDEs for each component
    kdes = [gaussian_kde(snrs[:,i]) for i in range(snrs.shape[1])]

    # Extract KDE values over interval 0 to 30
    x_range = np.linspace(0, 30, 10*30)  # 200 points from 0 to 30
    kde_values = np.zeros((len(kdes), len(x_range)))

    for i, kde in enumerate(kdes):
        kde_values[i] = kde(x_range)
        kde_values[i] /= kde_values[i].max()


    # Plot the kdes in a heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(kde_values, aspect='auto', extent=[0, 30, 1, snrs.shape[1]], origin='lower')
    plt.colorbar(label='Density')
    plt.xlabel('SNR')
    plt.ylabel('Principal Component')
    plt.title('KDE of SNRs by Principal Component')
    plt.tight_layout()

    # Plot average participant score SNRs
    plt.figure(figsize=(10, 5))

def run_cvpca_analysis():

    check_assignments = False
    recompute = False

    data = load_and_merge_raw_data(fmri_data_path, behavioral_data_path)
    X, flatcorrcols = get_functional_networks(data, power_csv_file, networks_to_inspect, check_assignments, short_names, recompute=recompute)

    # Perform consensus PCA
    cvpca = bootstrap_consensus_pca(X.values, n_bootstrap=100, alignment='correlation')

    # Plot the PCA results
    plot_consensus_loadings(cvpca)
    plot_variance_explained(cvpca)

    plot_pca_reliability(cvpca)
    plt.savefig('fig5ab.png', dpi=300)
    plt.close('all')

    # Save the CVPCA results to a file
    # with open("./data/cvpca.pickle", "wb") as file:
    #     pickle.dump(cvpca, file)

    plot_pc1_loadings(cvpca, flatcorrcols)
    plt.savefig('fig5c.png', dpi=300)

    # Load the CVPCA
    with open("./data/cvpca.pickle", "rb") as file:
        cvpca = pickle.load(file)

    # Transform the subject data 
    X = np.dot(X.copy(), cvpca['consensus_pcs'].T)
    X = pd.DataFrame(X, columns=[f'PC{i+1}' for i in range(X.shape[1])])

    X.to_csv("./data/functional_network_pc_scores.csv", index=False)


if __name__ == "__main__":
    run_cvpca_analysis()