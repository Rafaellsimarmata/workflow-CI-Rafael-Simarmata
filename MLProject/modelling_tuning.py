import os
import json
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import mlflow
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from dotenv import load_dotenv
load_dotenv()

mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URL')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('tuning.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('modelling_tuning.py')

def mlflow_setup():
    try:
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        else:
            raise ValueError('Dagrid_searchHub MLflow Tracking URI must set on the environment.')
        
        mlflow.set_experiment('Fraud Pred Tuned CI')
        logger.info('MLflow setup for Dagrid_searchHub completed.')
        
    except Exception as e:
        logger.exception(f'MLflow setup for Dagrid_searchHub failed: {e}.')
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        mlflow.set_experiment('Fraud Pred Tuned CI')
        logger.info('MLflow setup locally completed.')

def load_data(data_path = "fraud_detection_processed.csv"):
    logger.info(f'Loading data from: {data_path}')
    df = pd.read_csv(data_path)
    
    X = df.drop(['label'], axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Return the train-test split data
    logger.info(f'Data loaded and split. Train: {X_train.shape}, Test: {X_test.shape}')
    return X_train, X_test, y_train, y_test

def model_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),     # Overall correctness of the model
        'precision': precision_score(y_test, y_pred),   # Trustworthiness of "positive" predictions
        'recall': recall_score(y_test, y_pred),         # AKA sensitivity; How well the model identifies fraud
        'f1_score': f1_score(y_test, y_pred),           # Balance between precision and recall
        'cm_true_negative': tn,
        'cm_false_positive': fp,
        'cm_false_negative': fn,
        'cm_true_positive': tp,
        'tnr': tn / (tn + fp) if (tn + fp) != 0 else 0, # AKA specificity; How well the model identifies non-fraud ()
        'fnr': fn / (fn + tp) if (fn + tp) != 0 else 0, # AKA miss-rate; How well the model fails to identify fraud (track underdiagnosis)
        'fpr': fp / (fp + tn) if (fp + tn) != 0 else 0, # AKA fall-out; How well the model fails to identifies non-fraud as fraud (track overdiagnose fraud)
    }

    logger.info(f"Model {model} evaluated. Accuracy: {metrics['accuracy']:.4f}.")

    return metrics, cm


def xgboost_model_tuning(X_train, X_test, y_train, y_test):
    logger.info('Starting xgboost model tuning...')

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }

    xgb_clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, scoring='f1', cv=skf, verbose=2, n_jobs=-1)

    grid_search.fit(X_train, y_train)
    logger.info("Grid Search Xgboost completed.")

    logger.info("Best parameters found for xgboost: ", grid_search.best_params_)
    logger.info("Best F1 score found: ", grid_search.best_score_)

    best_params = grid_search.best_params_
    best_model = xgb.XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
    best_model.fit(X_train, y_train)

    metrics, cm = model_evaluate(best_model, X_test, y_test)
    logger.info(f"Xgboost model accuracy: {metrics['accuracy']:.4f}")

    with mlflow.start_run(run_name='xgb_tuned_run_ci') as run:
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        logger.info('Param logged to MLflow.')
        
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        logger.info('Metrics logged to MLflow.')

        cm_df = pd.DataFrame(
            cm, index=['Actual Negative', 'Actual Positive'],
            columns=['Predicted Negative', 'Predicted Positive']
        )

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_df.T, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for xgb Tuned', fontweight='bold', pad=10)
        plt.xlabel('Actual', fontweight='bold')
        plt.ylabel('Predicted', fontweight='bold')
        plt.tight_layout()
        cm_plot_path = 'models_tuned/xgb_tuned_confusion_matrix.png'
        os.makedirs(os.path.dirname(cm_plot_path), exist_ok=True)
        plt.savefig(cm_plot_path)
        mlflow.log_artifact(cm_plot_path)
        plt.close()
        logger.info('Confusion matrix logged as PNG to MLflow.')

        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values(by='importance', ascending=False)

        importance_path = 'models_tuned/xgb_tuned_feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        for feature, importance_value in zip(X_train.columns, best_model.feature_importances_):
            mlflow.log_param(f'importance_{feature}', importance_value)
        logger.info('Feature importance logged to MLflow.')

        metrics_cleaned = {
            k: float(v) if isinstance(v, np.floating)
            else int(v) if isinstance(v, np.integer)
            else v for k, v in metrics.items()
        }

        metrics_path = 'models_tuned/xgb_tuned_metrics.json'
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics_cleaned, f, indent=4)
        mlflow.log_artifact(metrics_path)
        logger.info('Metrics saved and logged to MLflow.')

        mlflow.sklearn.log_model(
            best_model, 'model',
            input_example=X_train.head(),
            signature=infer_signature(X_train, best_model.predict(X_train))
        )

        model_path = 'models_tuned/xgb_tuned_model.pkl'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)
        logger.info('Xgboost model logged to MLflow.')

        run_id = run.info.run_id
        logger.info('Xgboost model tuning completed.')

    return best_model, metrics, cm, run_id


def rf_model_tuning(X_train, X_test, y_train, y_test):
    logger.info('Starting Random Forest model tuning...')

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [25, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt']
    }

    rf = RandomForestClassifier(random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(rf, param_grid=param_grid, scoring='f1', cv=skf, verbose=2, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    logger.info(f'Random Forest best params: {best_params}')

    best_model = RandomForestClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    metrics, cm = model_evaluate(best_model, X_test, y_test)
    logger.info(f"Random Forest model accuracy: {metrics['accuracy']:.4f}")

    with mlflow.start_run(run_name='rf_tuned_run_ci') as run:
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        logger.info('Param logged to MLflow.')
        
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        logger.info('Metrics logged to MLflow.')

        cm_df = pd.DataFrame(
            cm, index=['Actual Negative', 'Actual Positive'],
            columns=['Predicted Negative', 'Predicted Positive']
        )

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_df.T, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for rf Tuned', fontweight='bold', pad=10)
        plt.xlabel('Actual', fontweight='bold')
        plt.ylabel('Predicted', fontweight='bold')
        plt.tight_layout()
        cm_plot_path = 'models_tuned/rf_tuned_confusion_matrix.png'
        os.makedirs(os.path.dirname(cm_plot_path), exist_ok=True)
        plt.savefig(cm_plot_path)
        mlflow.log_artifact(cm_plot_path)
        plt.close()
        logger.info('Confusion matrix logged as PNG to MLflow.')

        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values(by='importance', ascending=False)

        importance_path = 'models_tuned/rf_tuned_feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        for feature, importance_value in zip(X_train.columns, best_model.feature_importances_):
            mlflow.log_param(f'importance_{feature}', importance_value)
        logger.info('Feature importance logged to MLflow.')

        metrics_cleaned = {
            k: float(v) if isinstance(v, np.floating)
            else int(v) if isinstance(v, np.integer)
            else v for k, v in metrics.items()
        }

        metrics_path = 'models_tuned/rf_tuned_metrics.json'
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics_cleaned, f, indent=4)
        mlflow.log_artifact(metrics_path)
        logger.info('Metrics saved and logged to MLflow.')

        mlflow.sklearn.log_model(
            best_model, 'model',
            input_example=X_train.head(),
            signature=infer_signature(X_train, best_model.predict(X_train))
        )

        model_path = 'models_tuned/rf_tuned_model.pkl'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)
        logger.info('Random Forest model logged to MLflow.')

        run_id = run.info.run_id
        logger.info('Random Forest model tuning completed.')

    return best_model, metrics, cm, run_id

def adaboost_model_tuning(X_train, X_test, y_train, y_test):
    logger.info('Starting AdaBoost model tuning...')

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        'estimator': [
            DecisionTreeClassifier(max_depth=1),
            DecisionTreeClassifier(max_depth=3),
            DecisionTreeClassifier(max_depth=5)
        ]
    }

    ada = AdaBoostClassifier(random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(ada, param_grid=param_grid, scoring='f1', cv=skf, verbose=2, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    logger.info(f'AdaBoost best params: {best_params}')

    best_model = AdaBoostClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    metrics, cm = model_evaluate(best_model, X_test, y_test)
    logger.info(f"AdaBoost model accuracy: {metrics['accuracy']:.4f}")

    with mlflow.start_run(run_name='adaboost_tuned_run_ci') as run:
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        logger.info('Param logged to MLflow.')
        
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        logger.info('Metrics logged to MLflow.')

        cm_df = pd.DataFrame(
            cm, index=['Actual Negative', 'Actual Positive'],
            columns=['Predicted Negative', 'Predicted Positive']
        )

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_df.T, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for adaboost Tuned', fontweight='bold', pad=10)
        plt.xlabel('Actual', fontweight='bold')
        plt.ylabel('Predicted', fontweight='bold')
        plt.tight_layout()
        cm_plot_path = 'models_tuned/adaboost_tuned_confusion_matrix.png'
        os.makedirs(os.path.dirname(cm_plot_path), exist_ok=True)
        plt.savefig(cm_plot_path)
        mlflow.log_artifact(cm_plot_path)
        plt.close()
        logger.info('Confusion matrix logged as PNG to MLflow.')

        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values(by='importance', ascending=False)

        importance_path = 'models_tuned/adaboost_tuned_feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        for feature, importance_value in zip(X_train.columns, best_model.feature_importances_):
            mlflow.log_param(f'importance_{feature}', importance_value)
        logger.info('Feature importance logged to MLflow.')

        metrics_cleaned = {
            k: float(v) if isinstance(v, np.floating)
            else int(v) if isinstance(v, np.integer)
            else v for k, v in metrics.items()
        }

        metrics_path = 'models_tuned/adaboost_tuned_metrics.json'
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics_cleaned, f, indent=4)
        mlflow.log_artifact(metrics_path)
        logger.info('Metrics saved and logged to MLflow.')

        mlflow.sklearn.log_model(
            best_model, 'model',
            input_example=X_train.head(),
            signature=infer_signature(X_train, best_model.predict(X_train))
        )

        model_path = 'models_tuned/adaboost_tuned_model.pkl'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)
        logger.info('AdaBoost model logged to MLflow.')

        run_id = run.info.run_id
        logger.info('AdaBoost model tuning completed.')

    return best_model, metrics, cm, run_id

def main():
    try:
        mlflow_setup()

        X_train, X_test, y_train, y_test = load_data()

        results = {}

        rf_model, rf_metrics, rf_cm, rf_run_id = rf_model_tuning(X_train, X_test, y_train, y_test)
        results['random_forest'] = {
            'model': rf_model,
            'metrics': rf_metrics,
            'cm': rf_cm,
            'run_id': rf_run_id
        }

        adaboost_model, adaboost_metrics, adaboost_cm, adaboost_run_id = adaboost_model_tuning(X_train, X_test, y_train, y_test)
        results['adaptive_boosting'] = {
            'model': adaboost_model,
            'metrics': adaboost_metrics,
            'cm': adaboost_cm,
            'run_id': adaboost_run_id
        }

        xgb_model, xgb_metrics, xgb_cm, xgb_run_id = xgboost_model_tuning(X_train, X_test, y_train, y_test)
        results['decision_tree'] = {
            'model': xgb_model,
            'metrics': xgb_metrics,
            'cm': xgb_cm,
            'run_id': xgb_run_id
        }

        best_model_name = max(
            results.keys(),
            key=lambda k: (
                results[k]['metrics']['accuracy'],
                results[k]['metrics']['precision'],
                results[k]['metrics']['recall']
            )
        )
        best_model_info = results[best_model_name]

        logger.info(f'Best model: {best_model_name}')
        logger.info(f"Accuracy: {best_model_info['metrics']['accuracy']:.4f}")
        logger.info(f'Run ID: {best_model_info['run_id']}')

        run_id_path = 'models_tuned/best_tuned_model_run_id.txt'
        os.makedirs(os.path.dirname(run_id_path), exist_ok=True)
        with open(run_id_path, 'w') as f:
            f.write(best_model_info['run_id'])

        best_model_path = f'models_tuned/{best_model_name}_tuned_model.pkl'
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        joblib.dump(best_model_info['model'], best_model_path)

        logger.info(f'Best tuned model saved to: {best_model_path}')
        logger.info('All models tuned and logged successfully.')

        return results
    except Exception as e:
        logger.exception(f'An error occurred during model tuning: {e}')
        raise

if __name__ == '__main__':
    main()