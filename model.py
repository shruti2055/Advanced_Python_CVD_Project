"""Model Module.

Contains machine learning models and training functions.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_baseline_models():
    """
    Return baseline models (Logistic Regression and Random Forest).

    Returns
    -------
    dict
        Dictionary with baseline model names as keys and model instances as values.
    """
    baseline_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, 
                                               max_depth=20, min_samples_split=5, n_jobs=1)
    }
    return baseline_models


def get_tuned_models():
    """
    Return base models for tuning (XGBoost and LightGBM).

    Returns
    -------
    dict
        Dictionary with model names as keys and model instances as values.
    """
    tuned_base_models = {
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=1),
        'LightGBM': LGBMClassifier(random_state=42, verbosity=-1, n_jobs=1)
    }
    return tuned_base_models


def get_param_grids():
    """
    Return hyperparameter grids for tuning XGBoost and LightGBM.

    Returns
    -------
    dict
        Dictionary with model names as keys and parameter grids as values.
    """
    param_grids = {
        'XGBoost': {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, 8],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.5],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [1, 1.5, 2, 5]
        },
        'LightGBM': {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [-1, 5, 10, 15],
            'num_leaves': [20, 30, 40, 50],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_samples': [10, 20, 30],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [1, 1.5, 2, 5]
        }
    }
    return param_grids


def tune_xgboost_and_lightgbm(X_train, y_train):
    """
    Perform full hyperparameter tuning for XGBoost and LightGBM using RandomizedSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training target.

    Returns
    -------
    dict
        Dictionary with tuned XGBoost and LightGBM models.
    """
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING (XGBoost and LightGBM)")
    print("=" * 60)
    
    tuned_models = {}
    tuned_base_models = get_tuned_models()
    param_grids = get_param_grids()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name in ['XGBoost', 'LightGBM']:
        print(f"\n--- Tuning {name} ---")
        model = tuned_base_models[name]
        param_grid = param_grids[name]
        
        print(f"Search space: {len(param_grid)} parameters")
        print(f"Performing 20 random iterations...")
        
        search = RandomizedSearchCV(
            model, param_grid,
            n_iter=20,
            scoring='recall',
            cv=skf,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        search.fit(X_train, y_train)
        
        tuned_models[name] = search.best_estimator_
        print(f"Best recall score (CV): {search.best_score_:.4f}")
        print(f"Best hyperparameters detected")
    
    return tuned_models


def evaluate_all_models(baseline_models, tuned_models, X_train, y_train):
    """
    Evaluate all models (baseline + tuned) using cross-validation.

    Parameters
    ----------
    baseline_models : dict
        Dictionary of baseline models.
    tuned_models : dict
        Dictionary of tuned models.
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training target.

    Returns
    -------
    pd.DataFrame
        DataFrame with cross-validation results for all models.
    """
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION EVALUATION (All Models)")
    print("=" * 60)
    
    all_models = {**baseline_models, **tuned_models}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    
    results_list = []
    
    for name, model in all_models.items():
        print(f"\nEvaluating {name}...")
        scores = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring)
        
        result_row = {
            'Model': name,
            'Accuracy': f"{scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}",
            'Precision': f"{scores['test_precision'].mean():.4f} ± {scores['test_precision'].std():.4f}",
            'Recall': f"{scores['test_recall'].mean():.4f} ± {scores['test_recall'].std():.4f}",
            'F1-Score': f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}",
            'Recall_Mean': scores['test_recall'].mean(),
            'F1_Mean': scores['test_f1'].mean(),
            'Accuracy_Mean': scores['test_accuracy'].mean()
        }
        results_list.append(result_row)
    
    results_df = pd.DataFrame(results_list)
    return results_df


def select_best_model(results_df, all_models):
    """
    Select the best model based on recall, then F1-score, then accuracy.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with cross-validation results.
    all_models : dict
        Dictionary of all models.

    Returns
    -------
    tuple
        (best_model_name, best_model_instance)
    """
    print("\n" + "=" * 60)
    print("MODEL SELECTION")
    print("=" * 60)
    
    # Sort by recall (desc), then f1 (desc), then accuracy (desc)
    sorted_df = results_df.sort_values(
        by=['Recall_Mean', 'F1_Mean', 'Accuracy_Mean'],
        ascending=[False, False, False]
    )
    
    best_model_name = sorted_df.iloc[0]['Model']
    
    print(f"\n--- Ranking (sorted by Recall → F1 → Accuracy) ---")
    for idx, row in enumerate(sorted_df.itertuples(), 1):
        print(f"{idx}. {row.Model:20s} | Recall: {row.Recall_Mean:.4f} | F1: {row.F1_Mean:.4f}")
    
    print(f"\n✓ BEST MODEL: {best_model_name}")
    print(f"  - Recall: {sorted_df.iloc[0]['Recall_Mean']:.4f}")
    print(f"  - F1-Score: {sorted_df.iloc[0]['F1_Mean']:.4f}")
    print(f"  - Accuracy: {sorted_df.iloc[0]['Accuracy_Mean']:.4f}")
    
    best_model_instance = all_models[best_model_name]
    
    return best_model_name, best_model_instance


def train_model(model, X_train, y_train):
    """
    Train a model on the full training data.

    Parameters
    ----------
    model : sklearn estimator
        The model to train.
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training target.

    Returns
    -------
    trained model
        The trained model.
    """
    model.fit(X_train, y_train)
    return model


def get_feature_importance(model, feature_names, model_name):
    """
    Get feature importance if available.

    Parameters
    ----------
    model : trained model
        The trained model.
    feature_names : list
        List of feature names.
    model_name : str
        Name of the model.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with feature names and importance, or None if not available.
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return df_importance
    else:
        print(f"Feature importance not available for {model_name}")
        return None


def tune_classification_threshold(model, X_val, y_val, thresholds=[0.3, 0.4, 0.5, 0.6]):
    """
    Tune decision threshold for a classifier to optimize recall/F1.

    Parameters
    ----------
    model : trained model
        The trained classifier.
    X_val : pd.DataFrame or np.ndarray
        Validation features.
    y_val : pd.Series or np.ndarray
        Validation target.
    thresholds : list
        List of thresholds to test.

    Returns
    -------
    pd.DataFrame
        DataFrame with threshold tuning results.
    """
    y_proba = model.predict_proba(X_val)[:, 1]  # Probabilities for positive class
    
    threshold_results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        
        threshold_results.append({
            'Threshold': threshold,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
    
    return pd.DataFrame(threshold_results)


def get_top_features(importance_dfs, n_features=8, n_models=3):
    """
    Identify top features consistently important across tree-based models.

    Parameters
    ----------
    importance_dfs : dict
        Dictionary with model names and their importance DataFrames.
    n_features : int
        Number of top features to return.
    n_models : int
        Minimum number of models where feature should appear in top.

    Returns
    -------
    list
        List of selected feature names.
    """
    feature_ranks = {}
    
    for model_name, df in importance_dfs.items():
        for idx, row in df.iterrows():
            feature = row['feature']
            rank = idx + 1  # 1-indexed rank
            if feature not in feature_ranks:
                feature_ranks[feature] = []
            feature_ranks[feature].append(rank)
    
    # Average rank across models
    avg_ranks = {f: np.mean(ranks) for f, ranks in feature_ranks.items()}
    
    # Sort by average rank and select top
    selected = sorted(avg_ranks.items(), key=lambda x: x[1])[:n_features]
    selected_features = [f for f, _ in selected]
    
    return selected_features


def soft_voting_ensemble(models_dict, X_val, y_val, model_names=['Random Forest', 'XGBoost', 'LightGBM']):
    """
    Create soft voting ensemble from specified models.

    Parameters
    ----------
    models_dict : dict
        Dictionary of all trained models.
    X_val : pd.DataFrame or np.ndarray
        Validation features.
    y_val : pd.Series or np.ndarray
        Validation target.
    model_names : list
        Names of models to include in ensemble.

    Returns
    -------
    dict
        Predictions and probabilities from ensemble.
    """
    # Get probabilities from each model
    probabilities = []
    
    for name in model_names:
        if name in models_dict:
            model = models_dict[name]
            proba = model.predict_proba(X_val)[:, 1]
            probabilities.append(proba)
    
    # Average probabilities
    ensemble_proba = np.mean(probabilities, axis=0)
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_val, ensemble_pred)
    recall = recall_score(y_val, ensemble_pred)
    f1 = f1_score(y_val, ensemble_pred)
    accuracy = accuracy_score(y_val, ensemble_pred)
    
    return {
        'predictions': ensemble_pred,
        'probabilities': ensemble_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def compare_threshold_performances(model, X_test, y_test, thresholds=[0.3, 0.4, 0.5, 0.6]):
    """
    Compare model performance across multiple thresholds on test set.

    Parameters
    ----------
    model : trained model
        The trained classifier.
    X_test : pd.DataFrame or np.ndarray
        Test features.
    y_test : pd.Series or np.ndarray
        Test target.
    thresholds : list
        List of thresholds to test.

    Returns
    -------
    pd.DataFrame
        Performance metrics for each threshold.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        results.append({
            'Threshold': threshold,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        })
    
    return pd.DataFrame(results)