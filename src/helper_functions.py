import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import (roc_auc_score, f1_score, precision_recall_curve, 
                             auc, confusion_matrix, accuracy_score)
import matplotlib.pyplot as plt
import time
import datetime
from pathlib import Path

# ------------- STANDARD LOGISTIC REGRESSION -------------- outdated!
def fit_single_model(df, year, features):
    """
    Fits a logistic regression model for a specific year on training data.

    Input:
        df (DataFrame): training dataset
        year (integer): year for which we want to fit the model
        features (list of strings): covariates that should be included in the model
    Output:
        model: summary of the coefficients and metrics of the fitted model
    """
    # pick specific year from the dataset
    df_yr = df[df["event_year"] == year].copy()
    
    # if we pick invalid year, that isn't in the dataset:
    if df_yr.empty:
        return None
    
    # form the X matrix of covariates (add constant as well) and Y vector of responses
    X = sm.add_constant(df_yr[features])
    Y = df_yr["dropout"]
    
    # fit the model and return the summary
    try:
        model = sm.Logit(Y, X).fit(disp=0)
        return model
    except Exception as e:
        print(f"Year {year}: Optimization failed to converge.")
        return None
    
def fit_all_models(df, years, features, silent=False):
    """
    Fits several logistic models using function fit_single_model.

    Input:
        df (DataFrame): training dataset
        years (list of integers): years for which we want to fit the models
        features (list of strings): covariates that should be included in all the models
        silent (boolean): if True, suppresses the success print statements
    Output:
        models_dict: dictionary containing the fitted models, with years as keys
    """
    models_dict = {}
    
    # fit one year at a time
    for yr in years:
        model = fit_single_model(df, yr, features)
        if model is not None:
            # add model to the dictionary
            models_dict[yr] = model
            if not silent: print(f"Year {yr}: Training successful.")
        else:
            print(f"Year {yr}: Skipped due to lack of data or convergence failure.")
            
    return models_dict

def ensemble_prediction(models_dict, df_test, features):
    """
    Makes predictions on the test set for all models. 
    Then averages predictions from all models.

    Input:
        models_dict (dictionary): dictionary of models for several years
        df_test (parquet): testing set to predict on
        features (list of strings): covariates to be used in the regression prediction, 
            has to be the same as those used for training
    Output:
        ensemble_prob (array): probabilities of being classified as 1 after averaging all the predictions.
    """
    X_test = sm.add_constant(df_test[features])
    all_probs = []
    
    for yr, model in models_dict.items():
        # get probability P(Y=1 | X) for each model
        probs = model.predict(X_test)
        all_probs.append(probs)
    
    # mean of probabilities over all models for all the observations
    ensemble_prob = np.mean(all_probs, axis=0)

    return ensemble_prob

def district_cross_validation(df_train, years, features, plot=True, silent=True):
    """
    Performs "leave one district out cross-validation" by fitting logistic models
    on all districts except one and then predicting on the left-out one. 
    Metrics are calculated globally (pooled) across all out-of-fold predictions.
    
    Input: 
        df_train (DataFrame): training set 
        years (list of integers): years on which we fit the models
        features (list of strings): covariates for the linear model
        plot (boolean): if True, plots the precision-recall curve
        silent (boolean): if True, suppresses print statements
    Output:
        metrics (dictionary): global evaluation metrics
        successful_folds (integer): number of districts successfully run
    """
    districts = [d for d in df_train['home_district'].unique()]
    
    total_y_true = []
    total_y_prob = []
    
    # track how many districts successfully ran
    successful_folds = 0 

    for dist in districts:
        # split the data into training and validation
        train_fold = df_train[df_train['home_district'] != dist].copy()
        val_fold = df_train[df_train['home_district'] == dist].copy()
        
        # fit models for all years
        fold_ensemble = fit_all_models(train_fold, years, features, silent=silent)
        
        # safety net in case a model hasn't converged - break current iteration and continue
        if not fold_ensemble:
            if not silent:
                print(f"Warning: Fold for District {dist} skipped - no models converged.")
            continue

        # predict dropout probabilities on the validation fold
        probs = ensemble_prediction(fold_ensemble, val_fold, features)
        y_true = val_fold["dropout"]

        # store for plotting and pooled metrics
        total_y_true.extend(y_true)
        total_y_prob.extend(probs)
        
        # register that this fold worked
        successful_folds += 1

    # calculate metrics
    metrics = calculate_metrics(np.array(total_y_true), np.array(total_y_prob), silent=True)

    if plot:
        plot_pr_curve(np.array(total_y_true), np.array(total_y_prob))

    return metrics, successful_folds

# -------------- LOGISTIC REGRESSION WITH SPLINES ---------

def fit_single_model_ns(df, year, features, spline_features=None, df_spline=5):
    """
    Fits a logistic regression using natural cubic splines.
    
    Input:
        df (DataFrame): training dataset
        year (integer): year for which we want to fit the model
        features (list of strings): covariates for the linear part
        spline_features (list of strings): covariates to fit as natural splines
        df_spline (int): degrees of freedom for the splines
    Output:
        model: fitted statsmodels GLM/logit model object, or None if failed
    """
    df_yr = df[df["event_year"] == year].copy()
    if df_yr.empty: return None

    formula_parts = []

    linear_features = [f for f in features if f not in (spline_features or [])]
    for f in linear_features:
        formula_parts.append(f)
            
    if spline_features:
        for f in spline_features:
            formula_parts.append(f"cr({f}, df={df_spline}, constraints='center')")

    formula = "dropout ~ " + " + ".join(formula_parts)

    try:
        model = smf.logit(formula, data=df_yr).fit(disp=0)
        return model
    except Exception as e:
        print(f"Year {year}: {e}")
        return None

def fit_all_models_ns(df, years, features, spline_features=None, df_spline=5, silent=False):
    """
    Fits several logistic models using the fit_single_model_natural_splines function.

    Input:
        df (DataFrame): training dataset
        years (list of integers): years for which we want to fit the models
        features (list of strings): covariates that should be included in all the models
        spline_features (list of strings): covariates to be fitted as natural splines
        df_spline (int): degrees of freedom for the splines (5 = 3 internal knots)
        silent (boolean): if True, suppresses the success print statements
    Output:
        models_dict (dictionary): containing the fitted models, with years as keys, with number of observations stored
    """
    models_dict = {}
    
    # fit one year at a time
    for yr in years:
        model = fit_single_model_ns(
            df=df, 
            year=yr, 
            features=features, 
            spline_features=spline_features, 
            df_spline=df_spline
        )
        
        if model is not None:
            # we store a dictionary for each year containing the model and the  number of observations
            models_dict[yr] = {
                "model": model,
                "weight": model.nobs  
            }
            if not silent: print(f"Year {yr}: Training successful. (n = {model.nobs})")
        else:
            print(f"Year {yr}: Skipped due to lack of data or convergence failure.")
            
    return models_dict

def ensemble_prediction_ns(models_dict, df_test):
    """
    Makes predictions on the test (validation) set for all models. 
    Then averages predictions from all models.

    Input:
        models_dict (dictionary): dictionary of models for several years
        df_test (DataFrame): testing set to predict on
    Output:
        ensemble_results (DataFrame): table containing the event years, true responses and probability predictions
    """
    all_probs = []
    weights = []
    
    for yr, data in models_dict.items():
        # extract the model and the weight from the dictionary
        model = data["model"]
        weight = data["weight"]
        
        probs = model.predict(df_test)
        
        all_probs.append(probs)
        weights.append(weight)
    
    # np.average handles the weighted math for us automatically across all rows!
    ensemble_prob = np.average(all_probs, axis=0, weights=weights)

    ensemble_results = pd.DataFrame({
        'event_year': df_test['event_year'],
        'y_true': df_test['dropout'], 
        'y_prob': ensemble_prob
    })

    return ensemble_results

def district_cross_validation_ns(df_train, years, features, spline_features=None, df_spline=5, plot="yearly", silent=True):
    """
    Performs "leave one district out cross-validation" by fitting logistic models
    on all districts except one and then predicting on the left-out one. 

    Input: 
        df_train (DataFrame): training set 
        years (list of integers): years on which we fit the models
        features (list of strings): covariates for the linear part
        spline_features (list of strings): covariates to fit as natural splines
        df_spline (int): degrees of freedom for the splines
        plot (str): "yearly" to plot PR curves per year, "global" for overall, or None
        silent (boolean): if True, suppresses print statements
    Output:
        metrics (dictionary): global evaluation metrics extracted from the weighted average
        successful_folds (integer): number of districts successfully run
        metrics_table (DataFrame): detailed table of all metrics across all years
    """
    
    districts = [d for d in df_train['home_district'].unique()]
    
    all_fold_results = [] 
    successful_folds = 0 

    for dist in districts:
        # pick one district for validation and the rest is for training
        train_fold = df_train[df_train['home_district'] != dist].copy()
        val_fold = df_train[df_train['home_district'] == dist].copy()
        
        fold_ensemble = fit_all_models_ns(
            df=train_fold, 
            years=years, 
            features=features, 
            spline_features=spline_features, 
            df_spline=df_spline,
            silent=silent
        )
        
        if not fold_ensemble:
            print(f"Warning: Fold for District {dist} skipped - no models converged.")
            continue

        # produce ensemble results for this district
        ensemble_results = ensemble_prediction_ns(fold_ensemble, val_fold)
        # and store them with results from the other districts in a list (list of tables)
        all_fold_results.append(ensemble_results)
        
        successful_folds += 1

    # stack all the out-of-fold district predictions together into one big results table
    cv_results = pd.concat(all_fold_results, ignore_index=True)

    # calculate yearly metrics and extract the global weighted average
    metrics_table = calculate_metrics_by_year(cv_results, given_thresholds=None)
    metrics = metrics_table.loc['Global (Weighted)'].to_dict()

    # plot either the yearly PR-curves or one global PR-curve
    if plot == "yearly":
        plot_pr_curve_by_year(cv_results)
    elif plot == "global":
        plot_pr_curve(cv_results['y_true'], cv_results['y_prob'])

    return metrics, successful_folds, metrics_table

# -------------- PLOTS AND METRICS ------------------------

def get_coefficients_table(models_dict):
    """
    Extracts coefficients and their 95% confidence intervals into a DataFrame 
    to visualize how the effects and their uncertainty differ between years.
    
    Input:
        models_dict (dictionary): Dictionary containing the fitted models, with years as keys.
    Output:
        DataFrame: A table containing coefficient estimates and their upper/lower bounds indexed by year.
    """
    coef_data = []
    
    for yr, data in models_dict.items():
        model = data["model"] 
        
        # extract the coefficients and the 95% confidence intervals
        params = model.params
        ci = model.conf_int()
        
        # build a dictionary for each year
        yr_dict = {"year": yr}
        for param in params.index:
            yr_dict[f"{param}_coef"] = params[param]
            yr_dict[f"{param}_lower"] = ci.loc[param, 0]
            yr_dict[f"{param}_upper"] = ci.loc[param, 1]

        # save into a common DataFrame   
        coef_data.append(yr_dict)
    
    return pd.DataFrame(coef_data).set_index("year")

def find_threshold_pr(y_true, y_prob):
    """
    Finds the best threshold based on the precision-recall curve.
    
    Input:
        y_true (array): True binary labels (0 or 1).
        y_prob (array): Predicted probabilities.
    Output:
        threshold (float): Optimal probability threshold that maximizes the F1 score.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

def calculate_metrics(y_true, y_prob, given_threshold=None, silent=False):
    """
    Calculates multiple performance metrics for a set of predictions.

    Input: 
        y_true (array): true responses (0 or 1)
        y_prob (array): result of the ensemble prediction (array of values between 0 and 1)
        given_threshold (float, optional): cut-off probability between being classified as 0 or 1, if not given, it is calculated using find_threshold_pr
        silent (boolean): if set to False (default) the performance report is printed
    Output:
        metrics (dictionary): a dictionary containing ROC-AUC, PR-AUC, F1 score, Accuracy, etc.
    """
    if given_threshold is None:
        threshold = find_threshold_pr(y_true, y_prob)
    else:
        threshold = given_threshold
        
    y_pred = (y_prob > threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    no_info_rate = max(y_true.mean(), 1 - y_true.mean())
    
    metrics = {
        "Threshold": threshold,
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": auc(recall, precision),
        "Accuracy": accuracy_score(y_true, y_pred),
        "No Info Rate": no_info_rate,
        "Sensitivity (Recall)": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "Pos Pred Value (Precision)": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "Neg Pred Value": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "F1 Score": f1_score(y_true, y_pred)
    }

    if not silent:
        print("\n" + "="*40)
        print("      EXTENDED PERFORMANCE REPORT")
        print("="*40)
        for name, value in metrics.items():
            print(f"{name:<22}: {value:.4f}")
        print("="*40 + "\n")

    return metrics

def calculate_metrics_by_year(ensemble_results, given_thresholds=None, save_dir=None, file_name="model"):
    """
    Evaluates model performance both globally and on a year-by-year basis.
    
    Input:
        ensemble_results (DataFrame): DataFrame containing 'event_year', 'y_true', 'y_prob'.
        given_thresholds (dictionary, optional): dictionary of pre-determined optimal thresholds per year.
        save_dir (str, optional): directory to save the exported excel table.
        file_name (str, optional): custom filename identifier.
    Output:
        metrics_table (DataFrame): table of all metrics across all years.
    """
    all_metrics = {}
    weights = []

    # calculate metrics for each year separately
    years = sorted(ensemble_results['event_year'].unique())
    for yr in years:
        ensemble_yr = ensemble_results[ensemble_results['event_year'] == yr]
        
        yr_threshold = None
        # if a dictionary with yearly thresholds is given, use them 
        if given_thresholds is not None:
            yr_threshold = given_thresholds.get(yr)
        
        yr_metrics = calculate_metrics(
            y_true=ensemble_yr['y_true'], 
            y_prob=ensemble_yr['y_prob'], 
            given_threshold=yr_threshold, 
            silent=True
        )
        all_metrics[yr] = yr_metrics
        weights.append(len(ensemble_yr))
        
    metrics_table = pd.DataFrame(all_metrics).T

    # calculate a weighed average over all years
    weighted_avg = np.average(metrics_table, axis=0, weights=weights)
    metrics_table.loc['Global (Weighted)'] = weighted_avg
    # move it to the top of the table
    metrics_table = pd.concat([metrics_table.iloc[[-1]], metrics_table.iloc[:-1]])

    if save_dir:

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        file_path = Path(save_dir) / f"metrics_{file_name}.xlsx"
        metrics_table.to_excel(file_path)
    
        print(f"Metrics table successfully saved.")
    
    return metrics_table

def plot_pr_curve(y_true, y_prob):
    """
    Plots a global Precision-Recall curve along with the baseline 1 - No Information Rate (NIR).
    
    Input:
        y_true (array-like): true observed values (0 or 1).
        y_prob (array-like): predicted probabilities.
    Output:
        displays a matplotlib figure.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    baseline = y_true.mean()

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'PR AUC = {pr_auc:.4f}')
    plt.axhline(y=baseline, color='navy', linestyle='--', 
                label=f'1-NIR = {baseline:.4f}')
    plt.ylim(0, None)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Global Cross-Validation Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.show()

def plot_spline_effect(models_dict, df, feature, save_dir=None, file_name="feature"):
    """
    Plots the predicted probability for a specific feature fitted as a spline,
    holding all other variables constant at their median (numeric) or mode (categorical).
    
    Input:
        models_dict (dictionary): dictionary containing the fitted models.
        df (DataFrame): the dataset used to establish min/max ranges and median/mode baselines.
        feature (str): the name of the feature to plot.
        save_dir (str, optional): directory to save the plot image.
        file_name (str, optional): custom filename identifier.
    Output:
        displays a matplotlib figure and optionally saves it to disk.
    """
    # find the min and max of the feature we want to plot to define the x-axis of the plot
    x_min = df[feature].min()
    x_max = df[feature].max()
    x_vals = np.linspace(x_min, x_max, 100)
    
    # create the synthetic dataframe
    # fill in the values for the chosen feature
    synth_df = pd.DataFrame({feature: x_vals})
    
    # fill the rest of the features with constant baseline values
    for col in df.columns:
        # ignore these columns as they are not features in any model
        if col == feature or col in ["dropout", "event_year", "home_district", "active", "home_club"]:
            continue
        # numerical variables are kept at median
        elif pd.api.types.is_numeric_dtype(df[col]):
            synth_df[col] = df[col].median()
        # categorical variables are kept at mode
        else:
            synth_df[col] = df[col].mode()[0]
            

    fig = plt.figure(figsize=(8, 5))
    all_probs = []
    weights = []
    
    # loop through each year in the models dictionary and predict dropout status
    for yr, data in models_dict.items():
        model = data["model"]
        weight = data["weight"]
        
        probs = model.predict(synth_df)
        
        # plot individual splines as well
        plt.plot(x_vals, probs, color="#424141", linewidth=1, alpha=1)
        
        all_probs.append(probs)
        weights.append(weight)
    
    # form the ensemble probabilities
    ensemble_prob = np.average(all_probs, axis=0, weights=weights)

    # plot againts the synthetic x-values of the chosen feature    
    try:
        plt.plot(x_vals, ensemble_prob, linewidth=4, color="#2f0495")
    except Exception as e:
        print(f"Could not plot Year {yr}: {e}")

    # algorithm that dynamically cleans up feature names for plotting
    translations = {
        "num": "number of",
        "prev": "in the previous year"
    }
    words = feature.split("_")
    translated_words = [translations.get(w, w) for w in words]
    clean_feature_name = " ".join(translated_words)
    clean_feature_name = clean_feature_name.capitalize()
            
    #plt.title(f"Partial Effect of {feature} on Dropout Probability", fontsize=14)
    plt.ylim(0, 1)
    plt.xlabel(clean_feature_name, fontsize=16)
    plt.ylabel("Predicted probability of dropout", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # save figure in the given directory
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / f"spline_{file_name}_{feature}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved the spline plot!")

    plt.show()

def plot_coefficient_evolution(df_coefs, save_dir=None, file_name="model", n_cols=3):
    """
    Plots the evolution of model coefficients and their 95% confidence intervals over time.
    
    Input:
        df_coefs (DataFrame): DataFrame containing coefficients and CI bounds. 
                                 Must have columns ending in '_coef', '_lower', and '_upper'.
        save_dir (str, optional): directory to save the plot images.
        file_name (str, optional): custom filename identifier.
        n_cols (int): number of columns in the subplot grid.
    Output:
        displays a matplotlib figure and optionally saves it to disk.
    """
    # get the parameter names by looking for the '_coef' suffix
    base_params = [col.replace('_coef', '') for col in df_coefs.columns if col.endswith('_coef')]
    
    if not base_params:
        print("Error: No columns ending with '_coef' found in the dataframe.")
        return

    # plot sizes
    n_rows = (len(base_params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).flatten()

    for i, param in enumerate(base_params):
        # reconstruct the exact column names for this specific parameter
        coef_col = f"{param}_coef"
        lower_col = f"{param}_lower"
        upper_col = f"{param}_upper"
        
        # plot the coefficient value evolution
        axes[i].plot(df_coefs.index, df_coefs[coef_col], marker='o', linestyle='-', color="#f38748", label='Coefficient')
        
        # plot the shaded 95% confidence interval
        axes[i].fill_between(
            df_coefs.index, 
            df_coefs[lower_col], 
            df_coefs[upper_col], 
            color="#f38748", 
            alpha=0.2,
            label='95% CI'
        )
        
        # add title and axis labels to each plot
        axes[i].set_title(f"Effect of: {param}", fontweight="bold")
        axes[i].set_xlabel("Year", fontsize=12)
        if param == "C(sex)[T.M]":
            axes[i].set_ylabel(rf"$\beta$ for sex (M)", fontsize=12)
        else:
            axes[i].set_ylabel(rf"$\beta$ for {param}", fontsize=12)
        
        # plot the 0 line
        axes[i].axhline(0, color='black', linewidth=1, linestyle='--') 
        axes[i].grid(True, alpha=0.3)
        
        # add a legend to just the very first subplot
        if i == 0:
            axes[i].legend(loc='best')

    # remove any empty subplots at the bottom right of the grid
    for j in range(len(base_params), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / f"param_evolution_{file_name.lower()}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved the parameter plot!")

    plt.show()

def plot_pr_curve_by_year(ensemble_results, years_to_plot=None, save_dir=None, file_name="model"):
    """
    Plots a grid of individual Precision-Recall curves for each requested year, 
    followed by a joint plot of all curves overlapping.
    
    Input:
        ensemble_results (DataFrame): table containing 'event_year', 'y_true', and 'y_prob'.
        years_to_plot (list, optional): specific years to include in the plot.
        save_dir (str, optional): directory to save the plot images.
        file_name (str, optional): custom filename identifier.
    Output:
        displays matplotlib figures and optionally saves them to disk.
    """
    # determine which years to plot
    available_years = sorted(ensemble_results['event_year'].unique())
    if years_to_plot is None:
        test_years = available_years
    else:
        test_years = [y for y in years_to_plot if y in available_years]
        if not test_years:
            print(f"Error: None of the requested years {years_to_plot} were found.")
            print(f"Available years are: {available_years}")
            return
            
    # color map - possibly change!
    colors = plt.cm.plasma(np.linspace(0, 0.9, max(1, len(test_years))))

    # GRID OF INDIVIDUAL YEARLY PLOTS
    # create a grid, with shared x and y axes
    n_cols = 3
    n_rows = (len(test_years) + n_cols - 1) // n_cols
    fig_grid, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    
    # pick out the portion of the ensemble which corresponds to the given year
    for i, yr in enumerate(test_years):
        ensemble_yr = ensemble_results[ensemble_results['event_year'] == yr]
        y_true_yr = ensemble_yr['y_true']
        y_prob_yr = ensemble_yr['y_prob']
        
        precision, recall, _ = precision_recall_curve(y_true_yr, y_prob_yr)
        pr_auc = auc(recall, precision)
        baseline = y_true_yr.mean() # yearly 1-NIR
        
        # plot the intividual curve and the baseline
        axes[i].plot(recall, precision, color=colors[i], lw=2, label=f'AUC = {pr_auc:.4f}')
        axes[i].axhline(y=baseline, color='black', linestyle='--', linewidth=1.5, label=f'1-NIR = {baseline:.4f}')
        axes[i].set_title(f'Year: {yr}', fontweight='bold', fontsize=14)
        axes[i].set_xlim(0, 1.0)
        axes[i].set_ylim(0, 1.05)
        axes[i].grid(alpha=0.3)
        axes[i].legend(loc="upper right", fontsize=12)
        
        # add axis labels only to outer edges to keep it cleaner
        if i % n_cols == 0:
            axes[i].set_ylabel('Precision', fontsize=13)
        if i >= len(test_years) - n_cols:
            axes[i].set_xlabel('Recall', fontsize=13)

    # remove any empty subplots at the end of the grid
    for j in range(len(test_years), len(axes)):
        fig_grid.delaxes(axes[j])
        
    fig_grid.suptitle("Individual Precision-Recall Curves by Year", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # save the figure
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        grid_save_path = Path(save_dir) / f"pr_curve_grid_{file_name.lower()}.png"
        fig_grid.savefig(grid_save_path, dpi=300, bbox_inches='tight')
        print(f"Saved the grid plot!")
    
    plt.show()

    # THE JOINT PLOT WITH OVERLAPPING CURVES
    fig_joint = plt.figure(figsize=(10, 7))
    
    # once again, pick only the given year from the ensemble
    for i, yr in enumerate(test_years):
        ensemble_yr = ensemble_results[ensemble_results['event_year'] == yr]
        precision, recall, _ = precision_recall_curve(ensemble_yr['y_true'], ensemble_yr['y_prob'])
        pr_auc = auc(recall, precision)
        baseline = ensemble_yr['y_true'].mean()
        
        plt.plot(recall, precision, color=colors[i], lw=2, 
                 label=f'{yr} (AUC = {pr_auc:.4f} | 1-NIR = {baseline:.4f})')

    # add a global baseline for the joint ensemble
    global_baseline = ensemble_results['y_true'].mean()
    plt.axhline(y=global_baseline, color='black', linestyle='--', linewidth=1.5,
                label=f'Global 1-NIR = {global_baseline:.4f}')

    plt.ylim(0, 1.05)
    plt.xlim(0, 1.0)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    
    title_suffix = "All Years" if years_to_plot is None else f"Selected Years"
    plt.title(f'Joint Precision-Recall Curves ({title_suffix})', fontsize=14, fontweight='bold')
    
    plt.legend(title="Test Year", loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.grid(alpha=0.3)
    plt.tight_layout() 
    plt.show()

    # save the figure
    if save_dir:
        joint_save_path = Path(save_dir) / f"pr_curve_joint_{file_name.lower()}.png"
        fig_joint.savefig(joint_save_path, dpi=300, bbox_inches='tight')
        print(f"Saved the joint plot!")

# -------------- VARIABLE SELECTION -----------------------

def forward_feature_selection_ns(df_train, years, candidate_features, spline_candidates=None, df_spline=5):
    """
    Performs forward feature selection, starting with a null model, 
    always adding the feature that increases the PR-AUC the most, 
    stopping when addition no longer improves the score. At each step, 
    it chooes whether a feature should be modeled linearly or as a 
    natural spline to maximize pooled PR-AUC.
    
    Input:
        df_train (DataFrame): the training dataset containing the features and target variable.
        years (list of integers): the years on which to fit and cross-validate the models.
        candidate_features (list of strings): a list of all potential features to consider adding.
        spline_candidates (list of strings, optional): a subset of candidate features that can be tested as natural splines.
        df_spline (int): degrees of freedom to use if a feature is fitted as a spline.
    Output:
        selected_features (list of strings): the final list of selected features.
        selected_splines (list of strings): the subset of selected features modeled as splines.
        best_overall_pr_auc (float): the highest pooled PR-AUC achieved by the final model.
        history_df (DataFrame): a step-by-step record of the selection process.
    """
    
    if spline_candidates is None:
        spline_candidates = []
    
    selected_features = []
    selected_splines = []
    expected_folds = df_train['home_district'].nunique()
    history = []
        
    unselected_features = list(candidate_features)

    step = 1 

    total_start_time = time.time()

    # first fit the null model #TODO make this a bit more similar to the backward baseline layout
    metrics0, folds0 = district_cross_validation(df_train, years, [], plot=False)
    best_overall_pr_auc = metrics0["PR-AUC"]

    # fill in the first step in the history
    history.append({
                'Step': 0,
                'Feature_Added': "Baseline (Null)",
                'Type': "None",
                'PR_AUC': best_overall_pr_auc,
                'Total_Features': len(selected_features)
            })
    
    while unselected_features:
        print(f"\n--- Forward Selection: Step {step} ---")
        step_start_time = time.time()
        
        best_step_feature = None
        best_step_is_spline = False
        best_step_pr_auc = -1.0
        
        # used for timing
        features_to_evaluate = len(unselected_features)
        
        for feature in unselected_features:

            # ensure we don't use both types of peer variables: pick the better one and ignore the other 
            if "num_peers_prev" in selected_features:
                if "num_peers_prev20" in unselected_features:
                    unselected_features.remove("num_peers_prev20")
            if "num_peers_prev20" in selected_features:
                if "num_peers_prev" in unselected_features:
                    unselected_features.remove("num_peers_prev")


            # set of selected features plus loop through all unselected
            current_test_features = selected_features + [feature]
            
            # fit the models using the current features
            metrics_lin, folds_lin, _ = district_cross_validation_ns(
                df_train=df_train, years=years, 
                features=current_test_features, 
                spline_features=selected_splines, 
                df_spline=df_spline, plot=False, silent=True
            )
            
            # verify all folds ran and pull out the PR-AUC value
            if metrics_lin is not None and folds_lin == expected_folds:
                pr_auc_lin = metrics_lin['PR-AUC']
                # check if the current PR-AUC is better than the previous best value in this step
                if pr_auc_lin > best_step_pr_auc:
                    best_step_pr_auc = pr_auc_lin
                    best_step_feature = feature
                    best_step_is_spline = False
            
            # if the feature that is being checked is among spline candidates, 
            # it gets checked for spline effect too
            if feature in spline_candidates:
                current_test_splines = selected_splines + [feature]
                
                metrics_spl, folds_spl, _ = district_cross_validation_ns(
                    df_train=df_train, years=years, 
                    features=current_test_features, 
                    spline_features=current_test_splines, 
                    df_spline=df_spline, plot=False, silent=True
                )
                
                # verify all folds ran and pull out the PR-AUC value
                if metrics_spl is not None and folds_spl == expected_folds:
                    pr_auc_spl = metrics_spl['PR-AUC']
                    # check if the current PR-AUC is better than the previous best value in this step
                    if pr_auc_spl > best_step_pr_auc:
                        best_step_pr_auc = pr_auc_spl
                        best_step_feature = feature
                        best_step_is_spline = True

        # timing
        step_duration = time.time() - step_start_time
        time_per_eval = step_duration / features_to_evaluate        
        max_remaining_evals = (features_to_evaluate - 1) * features_to_evaluate / 2
        max_eta_seconds = time_per_eval * max_remaining_evals
        formatted_step_time = str(datetime.timedelta(seconds=int(step_duration)))
        formatted_max_eta = str(datetime.timedelta(seconds=int(max_eta_seconds)))
        
        print(f"Step {step} took {formatted_step_time}. Worst-case Max ETA remaining: {formatted_max_eta}")

        # check if the best PR-AUC value from this step is better than the overall best
        # if yes, set it as the new best value...
        if best_step_pr_auc > best_overall_pr_auc:
            best_overall_pr_auc = best_step_pr_auc
            # ...add the best feature from this step in the selected features...
            selected_features.append(best_step_feature)
            # ...and remove it from the variables that remain to be checked in the next steps.
            unselected_features.remove(best_step_feature)
            
            # do the same with the spline effects
            term_type = "Spline" if best_step_is_spline else "Linear"
            if best_step_is_spline:
                selected_splines.append(best_step_feature)
            
            # save history for plotting
            history.append({
                'Step': step,
                'Feature_Added': best_step_feature,
                'Type': term_type,
                'PR_AUC': best_overall_pr_auc,
                'Total_Features': len(selected_features)
            })
            
            print(f"Added '{best_step_feature}' ({term_type}) | New Best PR-AUC: {best_overall_pr_auc:.4f}")
            step += 1
        
        # if not, end the algorithm
        else:
            print(f"No improvement found. Stopping selection.")
            break
            
    # final report of results!
    total_duration = time.time() - total_start_time
    print("\n" + "="*50)
    print("      FINAL FORWARD FEATURE SELECTION REPORT")
    print("="*50)
    print(f"Total Time Taken: {str(datetime.timedelta(seconds=int(total_duration)))}")
    print(f"Total Selected Features ({len(selected_features)}): {selected_features}")
    print(f"Modeled as Splines ({len(selected_splines)}): {selected_splines}")
    print(f"Final Pooled PR-AUC: {best_overall_pr_auc:.4f}")
    print("="*50 + "\n")
    
    history_df = pd.DataFrame(history)
    
    return selected_features, selected_splines, best_overall_pr_auc, history_df

def backward_feature_selection_ns(df_train, years, candidate_features, spline_candidates=None, df_spline=5):
    """
    Performs backward feature selection. Starts with all features and 
    iteratively removes the one that increases the pooled PR-AUC the most, 
    stopping when removal no longer improves the score. At each step, 
    it chooses whether a feature should be modeled linearly or as a 
    natural spline to maximize pooled PR-AUC.
    
    Input:
        df_train (DataFrame): the training dataset containing the features and target variable.
        years (list of integers): the years on which to fit and cross-validate the models.
        candidate_features (list of strings): the initial full list of features to start the model with.
        spline_candidates (list of strings, optional): a subset of candidate features to be modeled as natural splines.
        df_spline (int): degrees of freedom to use for the spline features.
    Output:
        current_features (list of strings): the final list of features remaining after selection.
        current_splines (list of strings): the remaining features that are modeled as splines.
        best_overall_pr_auc (float): the highest pooled PR-AUC achieved by the final model.
        history_df (DataFrame): a step-by-step record of the feature removal process.
    """
    if spline_candidates is None:
        spline_candidates = []
        
    # start with all features
    current_features = list(candidate_features)
    current_splines = [f for f in current_features if f in spline_candidates]
    
    expected_folds = df_train['home_district'].nunique()
    history = []
    
    print("\n--- Backward Selection: Initial Baseline ---")
    step_start_time = time.time()
    
    # fit a model with everything for a baseline PR-AUC
    metrics_base, folds_base, _ = district_cross_validation_ns(
        df_train=df_train, years=years, 
        features=current_features, 
        spline_features=current_splines, 
        df_spline=df_spline, plot=False, silent=True
    )
    
    if metrics_base is not None and folds_base == expected_folds:
        best_overall_pr_auc = metrics_base['PR-AUC']
        print(f"Baseline PR-AUC with all {len(current_features)} features: {best_overall_pr_auc:.4f}")
    else:
        print("Error: The baseline model with all features failed to converge in all districts.")
        print("Try removing highly correlated features before starting backward selection.")
        return current_features, current_splines, -1.0, pd.DataFrame()
    
    history.append({
                'Step': 0,
                'Feature_Removed': "-",
                'PR_AUC': best_overall_pr_auc,
                'Remaining_Features': len(current_features)
            })

    step = 1
    total_start_time = time.time()
    
    while len(current_features) > 1:
        print(f"\n--- Backward Selection: Step {step} (Testing {len(current_features)} removals) ---")
        step_start_time = time.time()
        
        best_step_feature_to_remove = None
        best_step_pr_auc = -1.0
        
        features_to_evaluate = len(current_features)
        
        for feature in current_features:
            # remove the current feature/spline feature
            test_features = [f for f in current_features if f != feature]
            test_splines = [f for f in current_splines if f != feature]
            
            # fit the model with the remaining features
            metrics_test, folds_test, _ = district_cross_validation_ns(
                df_train=df_train, years=years, 
                features=test_features, 
                spline_features=test_splines, 
                df_spline=df_spline, plot=False, silent=True
            )
            
            # verify that all folds ran
            if metrics_test is not None and folds_test == expected_folds:
                pr_auc_test = metrics_test['PR-AUC']
                # check if the current PR-AUC value is better than the best value in this step
                # if it is, it becomes the best value
                if pr_auc_test > best_step_pr_auc:
                    best_step_pr_auc = pr_auc_test
                    best_step_feature_to_remove = feature
        
        # timing
        step_duration = time.time() - step_start_time
        time_per_eval = step_duration / features_to_evaluate
        max_remaining_evals = (features_to_evaluate - 1) * features_to_evaluate / 2
        max_eta_seconds = time_per_eval * max_remaining_evals       
        formatted_step_time = str(datetime.timedelta(seconds=int(step_duration)))
        formatted_max_eta = str(datetime.timedelta(seconds=int(max_eta_seconds)))
        
        print(f"Step {step} took {formatted_step_time}. Worst-case Max ETA remaining: {formatted_max_eta}")

        # check if the best PR-AUC value in this step is better than the over all best value
        if best_step_pr_auc > best_overall_pr_auc:
            # if yes, it becomes the best overall best value and the corresponding feature is removed
            best_overall_pr_auc = best_step_pr_auc
            current_features.remove(best_step_feature_to_remove)
            
            # it is also removed from splines
            if best_step_feature_to_remove in current_splines:
                current_splines.remove(best_step_feature_to_remove)
            
            # save history for plotting
            history.append({
                'Step': step,
                'Feature_Removed': best_step_feature_to_remove,
                'PR_AUC': best_overall_pr_auc,
                'Remaining_Features': len(current_features)
            })
            
            print(f"Removed '{best_step_feature_to_remove}' | New Best PR-AUC: {best_overall_pr_auc:.4f}")
            step += 1
        # if not, end the algorithm and report the results
        else:
            print(f"No removal improved the PR-AUC. Stopping selection.")
            break
            
    # final report!
    total_duration = time.time() - total_start_time
    print("\n" + "="*50)
    print("      FINAL BACKWARD FEATURE SELECTION REPORT")
    print("="*50)
    print(f"Total Time Taken: {str(datetime.timedelta(seconds=int(total_duration)))}")
    print(f"Total Selected Features ({len(current_features)}): {current_features}")
    print(f"Modeled as Splines ({len(current_splines)}): {current_splines}")
    print(f"Final Pooled PR-AUC: {best_overall_pr_auc:.4f}")
    print("="*50 + "\n")
    
    history_df = pd.DataFrame(history)
    
    return current_features, current_splines, best_overall_pr_auc, history_df