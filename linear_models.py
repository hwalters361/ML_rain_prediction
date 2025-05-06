import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import time

def prepare_data(data_path):
    """Load and prepare the dataset for linear models"""
    with np.load(data_path, allow_pickle=True) as data:
        train_videos = np.nan_to_num(data["train_images"])
        test_videos = np.nan_to_num(data["test_images"])
        train_labels = data["train_labels"]
        test_labels = data["test_labels"]
    
    # Reshape videos to 2D array (samples, features)
    n_train_samples = train_videos.shape[0]
    n_test_samples = test_videos.shape[0]
    
    train_videos_2d = train_videos.reshape(n_train_samples, -1)
    test_videos_2d = test_videos.reshape(n_test_samples, -1)
    
    return (train_videos_2d, train_labels), (test_videos_2d, test_labels)

def train_and_evaluate_models(train_data, train_labels, test_data, test_labels):
    """Train and evaluate both logistic and linear regression models"""
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        train_data, train_labels, test_size=0.2, random_state=42
    )
    
    # Initialize models
    logistic_model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    linear_model = LinearRegression()
    
    # Train and evaluate logistic regression
    print("Training Logistic Regression...")
    start_time = time()
    logistic_model.fit(X_train, y_train.ravel())
    train_time = time() - start_time
    
    # Evaluate logistic regression
    train_pred = logistic_model.predict(X_train)
    val_pred = logistic_model.predict(X_val)
    test_pred = logistic_model.predict(test_data)
    
    train_acc = accuracy_score(y_train.ravel(), train_pred)
    val_acc = accuracy_score(y_val.ravel(), val_pred)
    test_acc = accuracy_score(test_labels.ravel(), test_pred)
    
    print("\nLogistic Regression Results:")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Train and evaluate linear regression
    print("\nTraining Linear Regression...")
    start_time = time()
    linear_model.fit(X_train, y_train.ravel())
    train_time = time() - start_time
    
    # Evaluate linear regression
    train_pred = linear_model.predict(X_train).round().astype(int)
    val_pred = linear_model.predict(X_val).round().astype(int)
    test_pred = linear_model.predict(test_data).round().astype(int)
    
    train_acc = accuracy_score(y_train.ravel(), train_pred)
    val_acc = accuracy_score(y_val.ravel(), val_pred)
    test_acc = accuracy_score(test_labels.ravel(), test_pred)
    
    print("\nLinear Regression Results:")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    return logistic_model, linear_model

def plot_model_comparison(history_nn, logistic_acc, linear_acc):
    """Plot comparison of model performances"""
    plt.figure(figsize=(10, 6))
    
    # Plot neural network accuracy
    plt.plot(history_nn.training_accuracy, label='Neural Network (Training)', linestyle='--')
    plt.plot(history_nn.validation_accuracy, label='Neural Network (Validation)', linestyle='-')
    
    # Plot baseline models
    plt.axhline(y=logistic_acc, color='r', linestyle='--', label='Logistic Regression')
    plt.axhline(y=linear_acc, color='g', linestyle='--', label='Linear Regression')
    
    plt.title('Model Performance Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load and prepare data
    data_path = "data/cluster_sst.npz"
    (train_data, train_labels), (test_data, test_labels) = prepare_data(data_path)
    
    # Train and evaluate models
    logistic_model, linear_model = train_and_evaluate_models(
        train_data, train_labels, test_data, test_labels
    )
    
    # Note: To plot comparison with neural network, you'll need to pass the history object
    # from the neural network training. This can be done by importing it from RainPrediction8.py
    # or by running both models in sequence and passing the results. 