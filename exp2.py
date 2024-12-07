import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report)

# Paths
CSV_FILE = 'landmarks_features.csv'

# Load dataset
def load_dataset():
    data = pd.read_csv(CSV_FILE, header=None)
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0].apply(lambda x: os.path.basename(x).split('_')[0])
    return X, y

# Train and evaluate models
def evaluate_models(X, y):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'SVM': SVC(kernel='linear'),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=22)
    }

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Compute evaluation metrics
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, average='weighted') * 100
        recall = recall_score(y_test, y_pred, average='weighted') * 100
        f1 = f1_score(y_test, y_pred, average='weighted') * 100
        
        # Save results
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"Recall: {recall:.2f}%")
        print(f"F1 Score: {f1:.2f}%")

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=2))
    
    return results

# Main function
def main():
    X, y = load_dataset()
    results = evaluate_models(X, y)

    # Display final comparison
    print("\nModel Comparison Summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}%")

if __name__ == "__main__":
    main()
