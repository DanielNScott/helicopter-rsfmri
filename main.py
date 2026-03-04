from configs import *
from paths import *
from analysis import *
from plots import *
from dataio import *
from cvpca import *
from networks import *



def get_indep_and_dep_vars(indep=None, outcomes=None, snr_threshold=0.68, use_selections=False, exclusion_margin=None, forgetfulness_margin=None):
    '''Retrieve independent and dependent variables for analysis.'''

    # Default to using CVPCA results as predictors
    if indep is None:
        indep = 'CVPCA'

    if outcomes is None:
        outcomes = ['IOLike', 'RWLike', 'CPLike']

    # Load fMRI and behavioural data
    data, fmri_colnames = load_and_merge_raw_data(fmri_data_path, behavioral_data_path)
    n_original = len(data)

    # Exclude subjects near the IOWeight median boundary
    if exclusion_margin is not None:
        ioweight_median = data['IOWeight'].median()
        keep = (data['IOWeight'] - ioweight_median).abs() > exclusion_margin
        data = data.loc[keep]
        print(f"IOWeight exclusion: removed {(~keep).sum()} subjects within {exclusion_margin} of median ({ioweight_median:.3f}), {keep.sum()} remain")

    # Exclude high-IOWeight subjects near the forgetfulness boundary (~0.58)
    if forgetfulness_margin is not None:
        forget_boundary = 0.58
        near_boundary = (data['forgetfulness'] - forget_boundary).abs() <= forgetfulness_margin
        high_ioweight = data['IOWeight'] >= data['IOWeight'].median()
        exclude = near_boundary & high_ioweight
        data = data.loc[~exclude]
        print(f"Forgetfulness exclusion: removed {exclude.sum()} high-IOWeight subjects within {forgetfulness_margin} of boundary ({forget_boundary}), {(~exclude).sum()} remain")

    # Track which original rows survived exclusion, then reset index
    kept_rows = data.index.tolist()
    data = data.reset_index(drop=True)

    # Select which independent variable set to use in analysis
    if indep == 'Raw Correlation PCA':
        X = raw_data_pca(data, fmri_colnames)

    elif indep == 'CVPCA':
        # Load the consensus PCA scores
        X = pd.read_csv("./data/functional_network_pc_scores.csv")

        # Load the consensus PCA results
        cvpca = np.load("./data/cvpca.pickle", allow_pickle=True)

        # Keep PCs above the SNR percentile threshold
        score_snr = np.mean(np.abs(cvpca['consensus_scores']) / cvpca['consensus_scores_sds'], axis=0)
        cutoff = np.percentile(score_snr, snr_threshold * 100)
        X = X.loc[:, score_snr >= cutoff]

    elif indep == 'Network Aggregates':
        X, _ = get_functional_networks(data, power_csv_file, networks_to_inspect, None, short_names, recompute=False)

    elif indep == 'Raw Correlations':
        X = data.loc[:,fmri_colnames].copy()

    # Import and use the selections file
    selections = pd.read_csv(selections_file, index_col=0) if use_selections else None

    # Get the outcome variables
    Y = data.loc[:,outcomes].copy()

    # For predictors loaded from file, align rows with filtered data
    if len(kept_rows) < n_original and len(X) == n_original:
        X = X.iloc[kept_rows].reset_index(drop=True)

    return X, Y, outcomes, selections

def save_backward_elimination_results(X, fit_results, n_folds):
    # Aggregate the AIC and selection data
    predictors = X.columns.tolist()
    selection_counts, aic_means, aic_stds = aggregate_selection_and_aic_data(predictors, fit_results, cvtype='kfold')

    # Selected fields
    selections = (aic_means - 2*aic_stds/np.sqrt(n_folds) > 0)

    # Save to CSV files
    selections.to_csv(selections_file)
    selection_counts.to_csv(selection_counts_file)
    aic_means.to_csv(aic_means_file)
    aic_stds.to_csv(aic_stds_file)

def load_and_plot_selection_analysis(n_folds):
    # Load the selection data
    selection_counts = pd.read_csv(selection_counts_file, index_col=0)
    aic_means = pd.read_csv(aic_means_file, index_col=0)
    aic_stds = pd.read_csv(aic_stds_file, index_col=0)

    # Plot the selection analysis
    plot_selection_analysis(selection_counts, aic_means, aic_stds, n_folds)

def fit_prediction_models(X, Y, outcomes, selections=None, cvoptions=None, recovery_options=None, use_selections=False):

    if cvoptions is None:
        cvoptions = {
            'n_workers': 1,
            'algorithm': "regularized logistic regression",
            'do_backward_elim': False,
            'stop_fold': None,
            'null_ctrl': False,
            'copy_ctrl': False,
            'cvtypes': {'loocv': X.shape[0]}
        }

    if recovery_options is None:
        recovery_options = {
            'recovery_test': False,
            'recovery_flip': 0.1
        }

    # Initialize dictionaries for results and metrics
    fit_results, fit_metrics = {}, {}
    auc_mean, auc_sem = {}, {}

    # Loop over CV types
    for cvtype, n_folds in cvoptions['cvtypes'].items():
        
        # Initialize fit results and metrics for this CV
        fit_results[cvtype], fit_metrics[cvtype] = {}, {}

        # For each CV type, compute the CV for a given outcome variable
        for var in outcomes:

            # Select data
            X_selected = X.loc[:, selections[var].values].copy() if use_selections else X.copy()

            # If we're doing a recovery test, sample new binary outcomes
            Yvar = generate_recovery_data(X_selected, Y[var], recovery_options['recovery_flip']) if recovery_options['recovery_test'] else Y[var]

            # Train and cross-validate model
            print(f"Running cross-validated fitting for {var} using {cvtype}")
            fit_results[cvtype][var] = cross_validation(
                X_selected, Yvar,
                n_folds = n_folds, 
                null_ctrl = cvoptions['null_ctrl'], 
                copy_ctrl = cvoptions['copy_ctrl'], 
                algorithm = cvoptions['algorithm'], 
                n_workers = cvoptions['n_workers'], 
                do_backward_elim = cvoptions['do_backward_elim'],
                stop_fold = cvoptions['stop_fold']
            )

            if cvtype == 'kfold':
                # Get model metrics
                fit_metrics[cvtype][var] = get_aucs(fit_results[cvtype][var])

                # Get means and SEMs for each
                auc_mean[var] = np.mean(fit_metrics[cvtype][var]['test'])
                auc_sem[var] = np.std(fit_metrics[cvtype][var]['test']) / np.sqrt(n_folds)

    return fit_results, fit_metrics, auc_mean, auc_sem

def run_analysis_pipeline(indep=None, outcomes=None, cvoptions=None, recovery_options=None, use_selections=False, snr_threshold=0.68, exclusion_margin=None, forgetfulness_margin=None):

    # Prepare the data
    X, Y, outcomes, selections = get_indep_and_dep_vars(snr_threshold=snr_threshold, indep=indep, outcomes=outcomes, use_selections=use_selections, exclusion_margin=exclusion_margin, forgetfulness_margin=forgetfulness_margin)

    n_subj = X.shape[0]

    # Default cross-validation and recovery options
    if cvoptions is None:
        cvoptions = {
            'n_workers': 1,
            'algorithm': "regularized logistic regression",
            'do_backward_elim': False,
            'stop_fold': None,
            'null_ctrl': False,
            'copy_ctrl': False,
            'cvtypes': {'loocv': n_subj}
        }

    if recovery_options is None:
        recovery_options = {
            'recovery_test': False,
            'recovery_flip': 0.1
        }

    # Update LOOCV fold count to match actual subject count
    if 'loocv' in cvoptions['cvtypes']:
        cvoptions['cvtypes']['loocv'] = n_subj

    # Run the analysis
    fit_results, fit_metrics, auc_mean, auc_sem = fit_prediction_models(
        X, Y, outcomes, selections,
        cvoptions=cvoptions, recovery_options=recovery_options, use_selections=use_selections
    )

    # If we did backward elimination, save the generated data
    if cvoptions['do_backward_elim']:
        save_backward_elimination_results(X, fit_results, n_subj)

    if use_selections and not cvoptions['do_backward_elim']:
        load_and_plot_selection_analysis(n_subj)

    # Plot the results of our analyses
    if cvoptions['algorithm'] == 'linear regression':
        plot_loocv_regression(fit_results, outcomes)
    else:
        plot_loocvs_from_results_only(fit_results, outcomes)


def main(analysis='all'):

    # Figure 5a: raw correlation PCs, elastic net LOOCV
    if analysis in ('all', 'raw_pca'):
        run_analysis_pipeline(indep='Raw Correlation PCA', exclusion_margin=0.05, forgetfulness_margin=0.05)

    # Figure 5b: network aggregate correlations, elastic net LOOCV
    if analysis in ('all', 'network_agg'):
        run_analysis_pipeline(indep='Network Aggregates', exclusion_margin=0.05, forgetfulness_margin=0.05)

    # Figure 5c: consensus PCA of network aggregates, elastic net LOOCV
    if analysis in ('all', 'network_pca'):
        run_analysis_pipeline(indep='CVPCA', exclusion_margin=0.05, forgetfulness_margin=0.05)

    # Continuous regression: predict IOWeight from network aggregates
    if analysis in ('all', 'ioweight_regression'):
        cvoptions = {
            'n_workers': 1,
            'algorithm': "linear regression",
            'do_backward_elim': False,
            'stop_fold': None,
            'null_ctrl': False,
            'copy_ctrl': False,
            'cvtypes': {'loocv': None}
        }
        run_analysis_pipeline(indep='Network Aggregates', outcomes=['IOWeight'], exclusion_margin=0.05, cvoptions=cvoptions)

if __name__ == "__main__":
    main()