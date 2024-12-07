import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r".*SymbolDatabase.GetPrototype\(\) is deprecated.*")

# Paths
CSV_FILE = 'landmarks_features.csv'
MODEL_FILE = 'exercise_classifier.pkl'
SCALER_FILE = 'scaler.pkl'

# Load or train model
def load_or_train_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        print('Loaded existing model.')
    else:
        data = pd.read_csv(CSV_FILE, header=None)
        X = data.iloc[:, 1:]
        y = data.iloc[:, 0].apply(lambda x: os.path.basename(x).split('_')[0])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=22)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, average='weighted') * 100
        recall = recall_score(y_test, y_pred, average='weighted') * 100
        f1 = f1_score(y_test, y_pred, average='weighted') * 100

        print(f'Model Accuracy: {accuracy:.2f}%')
        print(f'Precision: {precision:.2f}%')
        print(f'Recall: {recall:.2f}%')
        print(f'F1 Score: {f1:.2f}%')

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=2))

        # Plot individual evaluation summaries
        plot_overall_metrics(y_test, y_pred)
        plot_per_class_metrics(y_test, y_pred, model.classes_)

        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        print('Trained and saved new model.')
    return model, scaler

# Function to plot overall metrics
def plot_overall_metrics(y_true, y_pred):
    scores = {
        'Accuracy': accuracy_score(y_true, y_pred) * 100,
        'Precision': precision_score(y_true, y_pred, average='weighted') * 100,
        'Recall': recall_score(y_true, y_pred, average='weighted') * 100,
        'F1 Score': f1_score(y_true, y_pred, average='weighted') * 100
    }
    metrics_df = pd.DataFrame(scores, index=['Score'])
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=metrics_df, palette='viridis')
    ax.set_title('Overall Metrics (%)')
    ax.set_ylim(0, 100)
    
    # Annotate the bars with values
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.tight_layout()
    plt.savefig('overall_metrics.png')
    plt.show()
    print('Overall metrics plot saved as overall_metrics.png')

# Function to plot per-class metrics
def plot_per_class_metrics(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    class_metrics = pd.DataFrame(report).transpose().iloc[:-3, :3] * 100  # Exclude average and support rows
    
    plt.figure(figsize=(20, 10))
    ax = class_metrics.plot(kind='bar', figsize=(16, 8))
    ax.set_title('Per Class Metrics (%)')
    ax.set_xlabel('Class')
    ax.set_ylabel('Percentage')
    ax.set_ylim(0,100)
    ax.legend(['Precision', 'Recall', 'F1 Score'], loc='upper right')
    
    # Annotate bars with values
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=6, color='black', xytext=(2, 5),
                    textcoords='offset points')

    plt.tight_layout()
    plt.savefig('per_class_metrics.png')
    plt.show()
    print('Per class metrics plot saved as per_class_metrics.png')

# Main function to run model training and evaluation
def main():
    model, scaler = load_or_train_model()

if __name__ == "__main__":
    main()
