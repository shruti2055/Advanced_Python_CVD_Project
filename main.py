"""Main pipeline.

Run the complete workflow for the cardiovascular disease prediction project.
"""

import pandas as pd
from data_manager import load_data, clean_data
from visualization import (
    run_all_visualizations,
    plot_confusion_matrix,
    plot_f1_comparison,
    plot_model_comparison,
    plot_threshold_tuning,
    plot_feature_importance
)
from preprocessing import preprocess_data
from model import (
    get_baseline_models,
    tune_xgboost_and_lightgbm,
    evaluate_all_models,
    select_best_model,
    train_model,
    get_feature_importance,
    get_top_features,
    soft_voting_ensemble,
    compare_threshold_performances
)
from evaluation import evaluate_model, compute_metrics, get_confusion_matrix


def main():
    """Run the project pipeline from loading data through final evaluation."""
    print("\n" + "=" * 60)
    print("CARDIOVASCULAR DISEASE PREDICTION PROJECT")
    print("=" * 60)

    data_path = r"C:\Users\Sanu\Documents\Advanced_Python\Project_CVD\cardio_train.csv"

    print("\nLoading data")
    raw_data = load_data(data_path)

    print("\nCleaning data")
    clean_data_frame = clean_data(raw_data)

    print("\nCreating exploratory plots")
    run_all_visualizations(clean_data_frame)

    print("\nPreprocessing data")
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(clean_data_frame)

    print("\nPreparing models")
    baseline_models = get_baseline_models()
    tuned_models = tune_xgboost_and_lightgbm(X_train, y_train)
    all_models = {**baseline_models, **tuned_models}

    print("\nEvaluating models by cross-validation")
    results_df = evaluate_all_models(baseline_models, tuned_models, X_train, y_train)
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))

    print("\nSelecting best model")
    best_model_name, best_model_instance = select_best_model(results_df, all_models)

    print(f"\nTraining best model: {best_model_name}")
    trained_best_model = train_model(best_model_instance, X_train, y_train)

    print("\nEvaluating best model on test data")
    test_metrics, y_pred = evaluate_model(trained_best_model, X_test, y_test)

    print("\nTuning decision threshold")
    thresholds = [0.3, 0.4, 0.5, 0.6]
    threshold_results = compare_threshold_performances(trained_best_model, X_test, y_test, thresholds)
    print(threshold_results.to_string(index=False))

    best_threshold_row = threshold_results.loc[threshold_results['Recall'].idxmax()]
    best_threshold = best_threshold_row['Threshold']
    y_pred_best_threshold = (trained_best_model.predict_proba(X_test)[:, 1] >= best_threshold).astype(int)
    threshold_metrics = compute_metrics(y_test, y_pred_best_threshold)

    print(f"\nBest threshold: {best_threshold}")
    print(f"Threshold recall: {threshold_metrics['recall']:.4f}")

    print("\nComputing feature importance for tree models")
    tree_models = {name: model for name, model in all_models.items() if name in ['Random Forest', 'XGBoost', 'LightGBM']}
    importance_dfs = {}

    for model_name, model in tree_models.items():
        fitted_model = train_model(model, X_train, y_train)
        importance_df = get_feature_importance(fitted_model, feature_names, model_name)
        if importance_df is not None:
            importance_dfs[model_name] = importance_df
            print(f"\n{model_name} top features")
            print(importance_df.head(5)[['feature', 'importance']].to_string(index=False))

    selected_features = get_top_features(importance_dfs, n_features=8)
    print(f"\nSelected features: {selected_features}")

    selected_indices = [feature_names.index(feature) for feature in selected_features if feature in feature_names]
    X_train_selected = X_train.iloc[:, selected_indices]
    X_test_selected = X_test.iloc[:, selected_indices]

    print("\nTraining best model with selected features")
    selected_best_model = train_model(best_model_instance, X_train_selected, y_train)
    test_metrics_selected, y_pred_selected = evaluate_model(selected_best_model, X_test_selected, y_test)

    print("\nBuilding voting ensemble")
    ensemble_models = {}
    for name in ['Random Forest', 'XGBoost', 'LightGBM']:
        if name in all_models:
            ensemble_models[name] = train_model(all_models[name], X_train, y_train)

    ensemble_results = soft_voting_ensemble(ensemble_models, X_test, y_test)
    print(f"Ensemble recall: {ensemble_results['recall']:.4f}")
    print(f"Ensemble F1-score: {ensemble_results['f1_score']:.4f}")

    comparison_data = {
        'Model': [
            'Baseline (Default Threshold)',
            f'Best Model + Threshold {best_threshold}',
            'Best Model + Feature Selection',
            'Ensemble (RF+XGB+LGBM)'
        ],
        'Recall': [
            test_metrics['recall'],
            threshold_metrics['recall'],
            test_metrics_selected['recall'],
            ensemble_results['recall']
        ],
        'F1-Score': [
            test_metrics['f1_score'],
            threshold_metrics['f1_score'],
            test_metrics_selected['f1_score'],
            ensemble_results['f1_score']
        ],
        'Precision': [
            test_metrics['precision'],
            threshold_metrics['precision'],
            test_metrics_selected['precision'],
            ensemble_results['precision']
        ],
        'Accuracy': [
            test_metrics['accuracy'],
            threshold_metrics['accuracy'],
            test_metrics_selected['accuracy'],
            ensemble_results['accuracy']
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    print("\nFinal model comparison")
    print(comparison_df.to_string(index=False))

    best_index = comparison_df['Recall'].idxmax()
    final_choice = comparison_df.iloc[best_index]
    print(f"\nFinal selected model: {final_choice['Model']}")

    if final_choice['Model'] == 'Baseline (Default Threshold)':
        final_predictions = y_pred
    elif final_choice['Model'] == f'Best Model + Threshold {best_threshold}':
        final_predictions = y_pred_best_threshold
    elif final_choice['Model'] == 'Best Model + Feature Selection':
        final_predictions = y_pred_selected
    else:
        final_predictions = ensemble_results['predictions']

    final_confusion = get_confusion_matrix(y_test, final_predictions)

    plot_confusion_matrix(final_confusion, final_choice['Model'])
    plot_model_comparison(results_df)
    plot_f1_comparison(results_df)
    plot_threshold_tuning(threshold_results, best_model_name)

    for name, importance_df in importance_dfs.items():
        plot_feature_importance(importance_df, name)

    print("\nPipeline complete")
    return {
        'best_model_name': best_model_name,
        'final_model': final_choice['Model'],
        'best_threshold': best_threshold,
        'baseline_results': test_metrics,
        'threshold_results': threshold_metrics,
        'feature_selection_results': test_metrics_selected,
        'ensemble_results': ensemble_results,
        'comparison': comparison_df,
        'final_choice': final_choice
    }


if __name__ == "__main__":
    results = main()
