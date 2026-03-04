import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

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
        text_offsets, text_colors = [5,0,-5], [corder[0], corder[1], corder[2]]
        text_offsets = [0.4, 0.2, 0.0]
        for i, var in enumerate(outcomes):
            y_test = fit_results['loocv'][var]["test_targs"]
            y_pred = fit_results['loocv'][var]["test_preds"]
            plot_LOOCV_AUCs(y_pred, y_test, text=True, text_offset=text_offsets[i], text_color=text_colors[i], newfig=False, label=var)
        plt.legend()
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

