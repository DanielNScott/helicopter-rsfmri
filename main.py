from xml.parsers.expat import model
from configs import *
from analysis import *
from plots import *
from dataio import *
from cvpca import *
from networks import *

# ---------------------------- Processing options -----------------------------#
n_workers = 1
algorithm = "regularized logistic regression"
do_backward_elim = False
snr_threshold = 1.0
use_network_aggregates = True
do_cvpca = False
use_cvpca = False
plot_cvpca = False
cvtypes = {"loocv":265}#{"kfold":10} #{'loocv':265, 'kfold':10}

# Use the cross-validation AIC selected predictors?
use_selections = False

# Debugging flags for cross-validation
copy_ctrl = False
null_ctrl = False
stop_fold = None
recovery_test = False
recovery_flip = 0.1

# File paths
fmri_data_path = "./data/subjX_withid.rds"
behavioral_data_path = "./data/subject_fits.csv"


# ---------------------------- Main script -----------------------------#
data, fmri_columns = load_data(fmri_data_path, behavioral_data_path)
n_subj = data.shape[0]
n_folds = n_subj
outcomes = ['IOLike', 'RWLike', 'CPLike']

# Raw data PCA
X = raw_data_pca(data, fmri_columns)

# Use network aggregated data?
if use_network_aggregates:
    X, flatcorrcols = get_functional_networks(data, power_csv_file, networks_to_inspect, check_assignments, short_names, recompute=recompute)
else:
    X = data.loc[:,fmri_columns]

# 
if use_cvpca:
    # Load the consensus PCA scores
    X = pd.read_csv("./data/functional_network_pc_scores.csv")

    # Load the consensus PCA results
    cvpca = np.load("./data/cvpca.pickle", allow_pickle=True)

    # Remove the low reliability PCs
    score_snr = np.mean(np.abs(cvpca['consensus_scores']) / cvpca['consensus_scores_sds'],axis=0)
    remove_idx = score_snr < snr_threshold
    X = X.loc[:, ~remove_idx]

# Import and use the selections file
if use_selections:
    selections = pd.read_csv(selections_file, index_col=0)


# Initialize dictionaries for results and metrics
fit_results, fit_metrics = {}, {}
auc_mean, auc_sem = {}, {}

# Loop over CV types

#cvtypes = {"kfold":10}
for cvtype, n_folds in cvtypes.items():
    
    # Initialize fit results and metrics for this CV
    fit_results[cvtype], fit_metrics[cvtype] = {}, {}

    # For each CV type, compute the CV for a given outcome variable
    for var in outcomes:

        # Select data
        X_selected = X.loc[:, selections[var].values].copy() if use_selections else X.copy()

        # If we're doing a recovery test, sample new binary outcomes
        Y = generate_recovery_data(X_selected, Y, recovery_flip) if recovery_test else data.loc[:,var]

        # Train and cross-validate model
        print(f"Running cross-validated fitting for {var} using {cvtype}")
        fit_results[cvtype][var] = cross_validation(
            X_selected, Y, 
            n_folds = n_folds, 
            null_ctrl = null_ctrl, 
            copy_ctrl = copy_ctrl, 
            algorithm = algorithm, 
            n_workers = n_workers, 
            do_backward_elim = do_backward_elim,
            stop_fold = stop_fold
        )

        if cvtype == 'kfold':
            # Get model metrics
            fit_metrics[cvtype][var] = get_aucs(fit_results[cvtype][var])

            # Get means and SEMs for each
            auc_mean[var] = np.mean(fit_metrics[cvtype][var]['test'])
            auc_sem[var] = np.std(fit_metrics[cvtype][var]['test']) / np.sqrt(n_folds)

# If we did backward elimination, save the generated data
if do_backward_elim:
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

if use_selections and not do_backward_elim:
    # Load the selection data
    selection_counts = pd.read_csv(selection_counts_file, index_col=0)
    aic_means = pd.read_csv(aic_means_file, index_col=0)
    aic_stds = pd.read_csv(aic_stds_file, index_col=0)

    # Plot the selection analysis
    plot_selection_analysis(selection_counts, aic_means, aic_stds, n_folds)

# Plot the results of our analyses
#plot_analysis_results(fit_results, auc_mean, auc_sem, outcomes)
plot_loocvs_from_results_only(fit_results, outcomes)
#plt.title('LOO Prediction ROC Curves: PCLR')
plt.title('Elastic Net, Raw Corr. PCs')
plt.savefig('fig5f.png', dpi=300)

# Extract all of the model p-values
if not do_backward_elim:
    # Get the regression coefficients from every fold
    coeffs_avg, coeffs_std, pvalues_avg, pvalues_std = extract_regression_results(fit_results, outcomes)
    plot_regression_stability(outcomes, use_selections, selections, flatcorrcols, coeffs_avg, coeffs_std)

    # Get the predictions from every fold
    n_folds = n_subj
    preds_avg, preds_std, preds_loo = extract_predictions(fit_results, n_subj, n_folds)

    # Plot the regression predictions
    plot_regression_predictions(X, data, preds_avg, preds_std, preds_loo)
