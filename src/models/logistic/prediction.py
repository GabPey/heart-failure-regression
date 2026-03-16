# We chose the model that we found the most balanced in inference criteria
# We will use it to predict the target variable.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from IPython.display import display

# We use the raw data for the first prediction model to see how it perfoms on prediction
def predict_with_raw_features(show_confusion_matrix=False):
    df = pd.read_csv("../data/raw/heart_failure_clinical_records_dataset.csv")
    X = df.drop(columns=["DEATH_EVENT"])
    y = df["DEATH_EVENT"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc_val = roc_auc_score(y_test, y_pred_proba)
    
    if show_confusion_matrix:
        from sklearn.metrics import confusion_matrix as cm_func
        cm = cm_func(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Vive", "Muere"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusión (Acc: {acc:.2f})")
        plt.show()
    
    results = {
        'model': pipeline,
        'accuracy': acc,
        'auc': auc_val,
        'classification_report': report,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba, 
        'y_pred': y_pred
    }
    
    return results

# We will use the model with engineered features for prediction, as it has better inference criteria.

def predict_with_inference_features(show_confusion_matrix=False):
    """
    Function that uses the variables identified in the inference model for data splitting and prediction.
    
    The inference model selected the following features:
    - constant (intercept)
    - time
    - age_centered
    - ejection_fraction_centered
    - sodium_creatinine_interaction 
    
    Returns:
    dict: Dictionary containing the trained model, predictions, and evaluation metrics.
    """
    # Load the processed data
    df = pd.read_csv("../data/processed/heart_failure_clinical_records_dataset_processed.csv")
    
    # Select the features identified in the inference model
    features = ["time", "age_centered", "ejection_fraction_centered", "sodium_creatinine_interaction"]
    X = df[features]
    y = df["DEATH_EVENT"]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Predicciones
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # MÉTRICAS
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc_val = roc_auc_score(y_test, y_pred_proba)
    
    if show_confusion_matrix:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Vive", "Muere"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusión (Acc: {acc:.2f})")
        plt.show()
    
    results = {
        'model': pipeline, # Guardamos el pipeline completo
        'accuracy': acc,
        'auc': auc_val,
        'classification_report': report,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba, 
        'y_pred': y_pred
    }
    
    return results

def display_model_performance(results):
    """
    Displays an organized summary of the model's performance metrics.
    Expects a 'results' dictionary containing: accuracy, auc, and classification_report.
    """
    # 1. Main Metrics Summary
    print("=" * 35)
    print("   MODEL PERFORMANCE SUMMARY")
    print("=" * 35)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"AUC-ROC:  {results['auc']:.4f}")
    print("-" * 35)
    
    # 2. Detailed Classification Report as a DataFrame
    report_df = pd.DataFrame(results['classification_report']).transpose()
    
    # Rename indices for better clarity (0=Survives, 1=Death)
    report_df.index = [
        "Lives (0)", 
        "Death (1)", 
        "Global Accuracy", 
        "Macro Avg", 
        "Weighted Avg"
    ]
    
    print("\nDETAILED CLASSIFICATION REPORT:")
    # Using styling to highlight key performance areas
    display(report_df.style.background_gradient(cmap='YlGnBu', subset=['precision', 'recall', 'f1-score']))