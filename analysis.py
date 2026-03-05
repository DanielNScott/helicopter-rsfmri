import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
import time
from multiprocessing import Pool
import multiprocessing as mp
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def raw_data_pca(data, fmri_columns):
    # Get the fmri columns of the raw data matrix
    X = data.loc[:,fmri_columns]

    # Perform the PCA
    pca = PCA()
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    pca.fit(X)
    X = pca.transform(X)
    X = pd.DataFrame(X, columns=[f'PC{i+1}' for i in range(X.shape[1])])

    # Select components explaining up to 50% variance
    ve_cumsum = np.cumsum(pca.explained_variance_ratio_)
    keep = np.arange(0,np.where(ve_cumsum >= 0.5)[0][0])
    X = X.iloc[:,keep]

    # Return the scores on these PCs
    return X

class ModelWrapper:
    def __init__(self, algorithm, backward_elim=False, balance_method='resample'):
        self.algorithm = algorithm
        self.balance_method = balance_method
        self.scaler = StandardScaler()
        self.p_values = None
        self.variables = None
        self.cumulative_variance = 0.9
        self.aic_impact = None
        self.standardize = True
        self.backward_elim = backward_elim

        # If backward elimination is on, set it as the fit method
        self.fit = self._fit if not backward_elim else self._backward_elimination

        # Logistic regression requires a constant column
        if self.algorithm in ['linear regression', 'logistic regression', 'pclr']:
            self.add_constant = True
        else:
            self.add_constant = False

    def preprocess(self, X):
        """Preprocessing of input data shared between fitting and predicting."""

        # Standardize the data using the mean and sd from fitting
        if self.standardize:
            X = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)

        # Apply any feature selection
        if self.variables and self.backward_elim:
            X = X.loc[:, self.variables] if isinstance(X, pd.DataFrame) else X[:, self.variables]

        # Add a constant column (whether classifiers do this differs)
        if self.add_constant:
            X = sm.add_constant(X, has_constant='add')

        return X

    def _fit(self,x, y):
        """Fit the model to the training data."""
        # Require x to be a pandas dataframe
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        # Fit the standardization transform
        if self.standardize:
            self.scaler.fit(x)

        # Standard preprocessing shared between fit and predict (DOES convert data to scores)
        #x_proc = x
        x_proc = self.preprocess(x)

        # Save the list of variables that end up in the model, excluding the constant
        self.variables = x_proc.columns.tolist()[1:] if self.add_constant else x_proc.columns.tolist()

        # Create the model instance
        if self.algorithm == "pclr":
            self.model = sm.Logit(y, x_proc).fit(disp=0)

        elif self.algorithm == "logistic regression":
            self.model = sm.Logit(y, x_proc).fit(disp=0)

        elif self.algorithm == "linear regression":
            self.model = sm.OLS(y, x_proc).fit(disp=0)  

        elif self.algorithm == "regularized logistic regression":
            cw = 'balanced' if self.balance_method == 'class_weight' else None
            self.model = LogisticRegression(l1_ratio=0.5, solver="saga", C=1, random_state=1112, max_iter=1000, class_weight=cw)
            self.model.fit(x_proc, y)

            self.model.params = self.model.coef_.ravel()
            self.model.pvalues = np.full(len(self.model.params), np.nan)

        elif self.algorithm == "naive bayes":
            self.model = GaussianNB()
            self.model.fit(x_proc, y)

        elif self.algorithm == "random forest":
            cw = 'balanced' if self.balance_method == 'class_weight' else None
            self.model = RandomForestClassifier(n_estimators=500, random_state=1112, class_weight=cw)
            self.model.fit(x_proc, y)

            self.model.params = self.model.feature_importances_
            self.model.pvalues = np.full(len(self.model.params), np.nan)

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Get p-values of everything except constant
        self.p_values = list(self.model.pvalues[1:]) if self.add_constant else list(self.model.pvalues)

    def predict(self, X):
        """Apply model to the input data to generate predictions."""
        # Raise error if X is not a pandas dataframe
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("Input data must be a pandas DataFrame or numpy array.")

        # Standard preprocessing shared between fit and predict
        X = self.preprocess(X)

        if self.algorithm in ["regularized logistic regression", "naive bayes", "random forest"]:
            preds = self.model.predict_proba(X)[:, 1]
        else:
            preds = self.model.predict(X)
        
        return preds
    
    def _get_aic(self):
        """Get AIC for the current model."""
        if hasattr(self.model, 'aic'):
            return self.model.aic
        else:
            # Calculate AIC manually for sklearn models
            # AIC = 2k - 2ln(L) where k=params, L=likelihood
            n_params = len(self.variables) + 1  # +1 for intercept if present
            
            if self.algorithm == "regularized logistic regression":
                # For logistic regression, calculate log likelihood
                y_pred_proba = self.model.predict_proba(self.X_fit)[:, 1]
                y_true = self.y_fit
                log_likelihood = np.sum(y_true * np.log(y_pred_proba + 1e-15) + 
                                    (1 - y_true) * np.log(1 - y_pred_proba + 1e-15))
            else:
                raise NotImplementedError(f"AIC calculation not implemented for {self.algorithm}")
            
            return 2 * n_params - 2 * log_likelihood

    def _backward_elimination(self, x, y):
        """Perform backward elimination based on AIC."""
        # Store data for AIC calculation
        self.y_fit = y
        
        # Do initial fit to establish full model
        self._fit(x, y)
        self.X_fit = self.preprocess(x)
        
        # Calculate AIC impact of each variable when all variables are present
        self._calculate_aic_impacts(x, y)
        
        continue_fitting = True
        while continue_fitting:
            current_aic = self._get_aic()
            
            # Try removing each variable and calculate AIC
            best_aic = current_aic
            best_var_to_remove = None
            
            for var in self.variables:
                temp_variables = [v for v in self.variables if v != var]
                
                if len(temp_variables) == 0:
                    continue

                # Temporarily set variables and fit model
                original_variables = self.variables.copy()
                self.variables = temp_variables

                try:
                    self._fit(x, y)
                    temp_aic = self._get_aic()

                    if temp_aic < best_aic:
                        best_aic = temp_aic
                        best_var_to_remove = var
                except:
                    pass

                # Restore original variables
                self.variables = original_variables

            # Remove the best variable if it improves AIC
            if best_var_to_remove is not None:
                self.variables.remove(best_var_to_remove)
            else:
                continue_fitting = False

        # Final fit with selected variables
        self._fit(x, y)

    def _calculate_aic_impacts(self, x, y):
        """Calculate the AIC impact of removing each variable from the full model."""
        # Get baseline AIC with all variables
        baseline_aic = self._get_aic()
        
        self.aic_impact = {}
        original_variables = self.variables.copy()
        
        for var in original_variables:
            # Create model without this variable
            temp_variables = [v for v in original_variables if v != var]
            
            if len(temp_variables) == 0:
                self.aic_impact[var] = np.inf  # Removing last variable = infinite impact
                continue
                
            # Temporarily fit without this variable
            self.variables = temp_variables
            
            try:
                self._fit(x, y)
                reduced_aic = self._get_aic()
                
                # Impact = AIC_without - AIC_with (positive = variable helps model)
                self.aic_impact[var] = reduced_aic - baseline_aic
                
            except:
                self.aic_impact[var] = np.inf  # Failed to fit without this variable
            
            # Restore original variables
            self.variables = original_variables
        
        # Restore the full model state
        self._fit(x, y)



def cross_validation(X, Y, n_folds=10, null_ctrl=False, copy_ctrl=False, n_workers=18, algorithm='pclr', do_backward_elim=True, stop_fold=None, balance_method='resample'):
    '''Perform cross-validation with parallel processing across folds.'''
    np.random.seed(1112)

    # Constrain n_workers to lesser of fold count and cpu cores.
    n_workers = min(n_folds, min(n_workers, mp.cpu_count()))

    # Create indices for k-fold cross-validation
    n_samples = X.shape[0]
    fold_indices = (np.arange(n_samples) % n_folds)
    
    # Shuffle fold assignments for k-fold (for LOOCV, each fold is one subject)
    if n_folds < n_samples:
        np.random.shuffle(fold_indices)

    # If checking null control, replace data with random noise
    if null_ctrl:
        X = pd.DataFrame(np.random.randn(X.shape[0], X.shape[1]), columns=X.columns, index=X.index)
        Y = pd.Series(np.random.randint(2, size=Y.shape[0]), index=Y.index, name=Y.name)

    # Prepare arguments for each fold
    fold_args = []
    for fold in range(n_folds):

        # Set up the arguments for this fold
        fold_args.append((X, Y, n_folds, fold_indices, fold, copy_ctrl, algorithm, do_backward_elim, balance_method))

        # Early stopping for debugging
        if stop_fold and (fold == stop_fold): break

    # Start parallel processing
    start_time = time.time()
    
    # Parallelization
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_single_fold, fold_args)
    else:
        results = []
        for args in fold_args:
            print(f"Processing fold {args[4]}")
            results.append(process_single_fold(args))

    # Display total runtime
    total_time = time.time() - start_time
    print(f"Total time with {n_workers} processes: {total_time:.2f} seconds")

    # Unpack results
    train_preds, train_targs, test_preds, test_targs, params, pvalues, aic_impacts, selected = [], [], [], [], [], [], [], []
    for i, (y_train, y_train_pred, y_test, y_test_pred, fold_params, fold_pvalues, fold_aic_impacts, fold_selected) in enumerate(results):
        train_preds.append(y_train_pred)
        train_targs.append(y_train)
        test_preds.append(y_test_pred)
        test_targs.append(y_test)
        params.append(fold_params)
        pvalues.append(fold_pvalues)
        aic_impacts.append(fold_aic_impacts)
        selected.append(fold_selected)

    # Convert to arrays, pack in dictionary, return
    return {
        "train_preds": train_preds,
        "train_targs": train_targs,
        "test_preds":  test_preds,
        "test_targs":  test_targs,
        "params":      params,
        "pvalues":     pvalues,
        "aic_impacts": aic_impacts,
        "selected":    selected
    }

def process_single_fold(args):
    """Process a single fold - designed to be called by worker processes."""
    X, Y, n_folds, fold_indices, fold, copy_ctrl, algorithm, do_backward_elim, balance_method = args

    # Create training and testing masks
    if n_folds > 1:
        train_mask = fold_indices != fold
        test_mask  = fold_indices == fold
    else:
        train_mask = fold_indices == fold
        test_mask  = fold_indices == fold

    # If checking copy control, replace test data with copy of training data
    if copy_ctrl:
        test_mask = train_mask

    # Training data and testing data
    x_train, y_train, x_test, y_test = X[train_mask].copy(), Y[train_mask].copy(), X[test_mask].copy(), Y[test_mask].copy()

    # Resample the training data to balance class membership (classification only)
    if algorithm not in ['linear regression'] and balance_method == 'resample':
        x_train, y_train = get_balanced_resample(x_train, y_train)

    # Train and test the model using consolidated function
    model = ModelWrapper(algorithm=algorithm, backward_elim=do_backward_elim, balance_method=balance_method)
    model.fit(x_train, y_train)

    aic_impacts = model.aic_impact
    selected = model.variables

    # Model parameters
    params  = model.model.params
    pvalues = model.model.pvalues

    # Get training and testing predictions
    y_train_pred = model.predict(x_train)
    y_test_pred  = model.predict(x_test)

    return y_train, y_train_pred, y_test, y_test_pred, params, pvalues, aic_impacts, selected

def tune_log_reg_parameters(data, fmri_columns, outcome, subset_fraction=0.3, cv_folds=5):
    """
    Tune C and l1_ratio parameters using cross-validation on a subset of data.
    
    Parameters:
    - subset_fraction: fraction of data to use for parameter tuning
    - cv_folds: number of CV folds for parameter selection
    
    Returns:
    - best_C: optimal regularization parameter
    - best_l1_ratio: optimal L1/L2 mixing parameter
    """
    
    # Create subset of data for parameter tuning
    np.random.seed(1112)
    n_subset = int(data.shape[0] * subset_fraction)
    subset_indices = np.random.choice(data.shape[0], n_subset, replace=False)
    subset_data = data.iloc[subset_indices]
    
    # Prepare data
    X_subset = subset_data[fmri_columns]
    y_subset = subset_data[outcome].values
    
    # Standardize
    scaler = StandardScaler()
    X_subset_scaled = scaler.fit_transform(X_subset)
    
    # Define parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    
    # Create model
    model = LogisticRegression(
        solver='saga', 
        random_state=1112,
        max_iter=1000
    )
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=cv_folds, 
        scoring='roc_auc',
        n_jobs=-1  # Use all available cores
    )
    
    grid_search.fit(X_subset_scaled, y_subset)
    
    print(f"Best parameters: C={grid_search.best_params_['C']}, l1_ratio={grid_search.best_params_['l1_ratio']}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_['C'], grid_search.best_params_['l1_ratio']


def get_aucs(fit_results):
    '''Returns row-wise prediction quality statistics.'''
    # Initialize lists to store results
    aucs = { "train": [], "test": []}

    # Get number of folds
    n_folds = len(fit_results['train_preds'])
    
    # Compute AUCs
    for group in ['train', 'test']:
        for fold in np.arange(n_folds):
            fpr, tpr, _ = roc_curve(fit_results[group+'_targs'][fold], fit_results[group+'_preds'][fold])
            aucs[group].append(auc(fpr, tpr))

        # Stack into arrays
        aucs[group] = np.stack(aucs[group])

    return aucs


def get_loocv_regression_stats(fit_results):
    """Compute R-squared, correlation, and p-value from LOOCV predictions."""
    from scipy.stats import pearsonr
    test_preds = np.concatenate(fit_results['test_preds'])
    test_targs = np.concatenate(fit_results['test_targs'])

    # Correlation and p-value
    r, p = pearsonr(test_preds, test_targs)

    # R-squared (proportion of variance explained by CV predictions)
    ss_res = np.sum((test_targs - test_preds) ** 2)
    ss_tot = np.sum((test_targs - np.mean(test_targs)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return {'r': r, 'r_squared': r_squared, 'p': p}


def get_balanced_resample(X, Y, remove=False):

    # Enforce indices being equal
    if isinstance(X, pd.DataFrame) and isinstance(Y, pd.Series):
        # Require X and y to have the same index sets
        assert X.index.equals(Y.index), "X and Y must have the same indices."

    # Compute the number of positive and negative samples
    n_pos = np.sum(Y == 1)
    n_neg = np.sum(Y == 0)

    # Determine which class is in excess
    excess = "positive" if n_pos > n_neg else "negative"

    # Compute the number of samples to generate
    n_large = np.max([n_pos, n_neg])
    n_small = np.min([n_pos, n_neg])
    n_excess = n_large - n_small

    # Either remove rows or resample to increase
    if not remove:
        # Get indices of the minority class
        idx = np.where(Y==0)[0] if excess == "positive" else np.where(Y==1)[0]

        # Generate indices for new rows to include
        new_rows = np.random.choice(idx, n_excess, replace=True)

        # Resample the data
        X = pd.concat([X, X.iloc[new_rows]],axis=0).reset_index(drop=True)
        Y = pd.concat([Y, Y.iloc[new_rows]],axis=0).reset_index(drop=True)
    else:
        # Randomly sample indices to remove from the majority class
        idx = np.where(Y==1)[0] if excess == "positive" else np.where(Y==0)[0]

        # Set rows to remove
        remove_rows = np.random.choice(idx, n_excess, replace=False)

        # Remove rows from the majority class (resetting indices to positional ones first)
        X = X.reset_index(drop=True).drop(remove_rows).reset_index(drop=True)
        Y = Y.reset_index(drop=True).drop(remove_rows).reset_index(drop=True)

    return X, Y


def fit_logistic_regression(X,Y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = sm.add_constant(X_scaled)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns.insert(0, 'const'))
    model = sm.Logit(Y, X_scaled).fit(disp=0)
    params = model.params
    pvals = model.pvalues
    y_hat = model.predict(X_scaled)
    return params, pvals, y_hat, model

def fit_regularized_logistic_regression(X,Y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    #X_scaled = sm.add_constant(X_scaled)
    #X_scaled = pd.DataFrame(X_scaled, columns=X.columns.insert(0, 'const'))
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    model = LogisticRegression(l1_ratio=0.5, solver="saga", C=1, random_state=1112, max_iter=1000)
    model.fit(X_scaled, Y)


    #model = sm.Logit(Y, X_scaled).fit(disp=0)
    params = model.params
    pvals = model.pvalues
    y_hat = model.predict(X_scaled)
    return params, pvals, y_hat, model


def aggregate_selection_and_aic_data(predictors, fit_results, cvtype):
    # Aggregate selection counts and AIC impacts into DataFrames
    selected_predictors = {}
    outcomes = ['IOLike', 'RWLike', 'CPLike']
    for var in outcomes:
        selected_predictors[var] = fit_results[cvtype][var]['selected']

    # Count number of times each predictor was selected
    selection_counts = {}
    for outcome in selected_predictors:
        selection_counts[outcome] = {}
        for variable in predictors:
            count = sum(1 for fold_vars in selected_predictors[outcome] if variable in fold_vars)
            selection_counts[outcome][variable] = count

    selection_counts = pd.DataFrame(selection_counts)

    # Get the AICs
    aic_aggregated = {}
    for outcome in fit_results[cvtype]:
        if 'aic_impacts' in fit_results[cvtype][outcome]:
            aic_impacts_list = fit_results[cvtype][outcome]['aic_impacts']
            
            # Collect all impacts for each variable across folds
            outcome_impacts = {}
            for fold_impacts in aic_impacts_list:
                for var, impact in fold_impacts.items():
                    if var not in outcome_impacts:
                        outcome_impacts[var] = []
                    outcome_impacts[var].append(impact)
            
            aic_aggregated[outcome] = outcome_impacts


    # Convert to DataFrame for easier handling
    aic_means = {}
    aic_stds = {}
    for outcome in aic_aggregated:
        aic_means[outcome] = {var: np.mean(impacts) for var, impacts in aic_aggregated[outcome].items()}
        aic_stds[outcome] = {var: np.std(impacts) for var, impacts in aic_aggregated[outcome].items()}

    aic_means = pd.DataFrame(aic_means)
    aic_stds = pd.DataFrame(aic_stds)

    return selection_counts, aic_means, aic_stds


def generate_recovery_data(X, Y, recovery_flip):
    """Generate synthetic binary outcomes from fitted model predictions."""
    n_subj = X.shape[0]

    # Generate recovery data using the fit model
    _, _, y_hat, _ = fit_logistic_regression(X, Y)
    Y_new = (y_hat > np.median(y_hat)).astype(int)

    # Flip a fraction of outcomes to prevent perfect separation
    flip_indices = np.random.choice(n_subj, size=int(n_subj * recovery_flip), replace=False)
    Y_new[flip_indices] = 1 - Y_new[flip_indices]
    Y_new = Y_new.reset_index(drop=True)

    return Y_new

def extract_regression_results(fit_results, outcomes):
    coeffs, pvalues, coeffs_avg, coeffs_std, pvalues_avg, pvalues_std = {}, {}, {}, {}, {}, {}
    for var in outcomes:
        coeffs[var]  = pd.concat(fit_results['loocv'][var]['params'], axis=1)
        pvalues[var] = pd.concat(fit_results['loocv'][var]['pvalues'], axis=1)

        # Average coefficients
        coeffs_avg[var] = coeffs[var].mean(axis=1)
        coeffs_std[var] = coeffs[var].std(axis=1)

        pvalues_avg[var] = pvalues[var].mean(axis=1)
        pvalues_std[var] = pvalues[var].std(axis=1)

    return coeffs_avg, coeffs_std, pvalues_avg, pvalues_std


def extract_predictions(fit_results, n_subj, n_folds):
    # Extract predictions from LOOCV results
    preds_avg, preds_std, preds_loo = {}, {}, {}
    for var in ['IOLike', 'RWLike', 'CPLike']:
        predictions = np.full((n_subj, n_folds), np.nan)
        loos = np.zeros((n_folds))
        all_indices = np.arange(n_subj)
        for fold in range(n_folds):
            fold_mask = all_indices != fold
            predictions[fold_mask,fold] = fit_results['loocv'][var]['train_preds'][fold].values[0:n_subj-1]
            loos[fold] = fit_results['loocv'][var]['test_preds'][fold].values[0]

        # Average predictions across folds
        preds_avg[var] = np.nanmean(predictions, axis=1)
        preds_std[var] = np.nanstd(predictions, axis=1)
        preds_loo[var] = np.array(loos)

    return preds_avg, preds_std, preds_loo