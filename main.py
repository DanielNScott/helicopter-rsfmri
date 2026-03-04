from configs import *
from paths import *
from analysis import *
from plots import *
from dataio import *
from cvpca import *
from networks import *



def get_indep_and_dep_vars(indep=None, snr_threshold=1.0, use_selections=False):
    '''Retrieve independent and dependent variables for analysis.'''

    # Default to using CVPCA results as predictors
    if indep is None:
        indep = 'CVPCA'

    # Load fMRI and behavioural data
    data, fmri_colnames = load_and_merge_raw_data(fmri_data_path, behavioral_data_path)
    outcomes = ['IOLike', 'RWLike', 'CPLike']

    # Select which independent variable set to use in analysis
    if indep == 'Raw Correlation PCA':
        X = raw_data_pca(data, fmri_colnames)

    elif indep == 'CVPCA':
        # Load the consensus PCA scores
        X = pd.read_csv("./data/functional_network_pc_scores.csv")

        # Load the consensus PCA results
        cvpca = np.load("./data/cvpca.pickle", allow_pickle=True)

        # Remove the low reliability PCs
        score_snr = np.mean(np.abs(cvpca['consensus_scores']) / cvpca['consensus_scores_sds'],axis=0)
        remove_idx = score_snr < snr_threshold
        X = X.loc[:, ~remove_idx]

    elif indep == 'Network Aggregates':
        X, _ = get_functional_networks(data, power_csv_file, networks_to_inspect, None, short_names, recompute=False)

    elif indep == 'Raw Correlations':
        X = data.loc[:,fmri_colnames].copy()

    # Import and use the selections file
    selections = pd.read_csv(selections_file, index_col=0) if use_selections else None

    # Get the outcome variables
    Y = data.loc[:,outcomes].copy()

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
            'cvtypes': {'loocv':265, 'kfold':10}
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

    #cvtypes = {"kfold":10}
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

def main():
    # Use the cross-validation AIC selected predictors?
    use_selections = False

    # Cross validation options
    cvoptions = {
        'n_workers': 1,
        'algorithm': "regularized logistic regression",
        'do_backward_elim': False,
        'stop_fold': None,
        'null_ctrl': False,
        'copy_ctrl': False,
        'cvtypes': {'loocv':265, 'kfold':10}
    }

    # Recovery options
    recovery_options = {
        'recovery_test': False,
        'recovery_flip': 0.1
    }

    n_subj = 265


    # Prepare the data
    X, Y, outcomes, selections = get_indep_and_dep_vars(use_selections=use_selections)

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
    plot_loocvs_from_results_only(fit_results, outcomes)
    plt.title('Elastic Net, Raw Corr. PCs')
    plt.savefig('fig5f.png', dpi=300)



if __name__ == "__main__":
    main()