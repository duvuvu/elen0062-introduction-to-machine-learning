"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 2 - Bias and variance analysis
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from data import load_wine_quality

def experiment(model, sample_sizes, X_pool, y_pool, X_test, y_test, M=50):
    """Run the experiment to evaluate the effect of training sample size on bias and variance."""
    expected_errors = []
    model_variances = []
    biases_plus_residuals = []

    for N in sample_sizes:
        predictions = []

        for _ in range(M):
            X_train, _, y_train, _ = train_test_split(X_pool, y_pool, train_size=N, 
                                                      random_state=None)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions.append(y_pred)

        predictions = np.array(predictions)

        expected_error = np.mean((predictions - y_test) ** 2)
        expected_errors.append(expected_error)

        variance = np.mean(np.var(predictions, axis=0))
        model_variances.append(variance)

        bias_plus_residual = expected_error - variance
        biases_plus_residuals.append(bias_plus_residual)

    return sample_sizes, expected_errors, model_variances, biases_plus_residuals

def plot_complexity_curve(fname, complexity_params,
                          expected_error, model_variance,
                          residual_bias):
    try:
        plt.figure(figsize=(5, 5))
        plt.plot(complexity_params, expected_error, label="Expected Error",
                 marker='o', markersize=5, alpha=0.8)
        plt.plot(complexity_params, residual_bias, label="Bias + Residual Error",
                 marker='o', markersize=5, alpha=0.8)
        plt.plot(complexity_params, model_variance, label="Variance", marker='o',
                 markersize=5, alpha=0.8)
        plt.xscale('log')
        plt.xlabel("N")
        plt.ylabel("Error")
        plt.xticks(complexity_params)
        plt.legend()
        plt.savefig(fname)
    except Exception as e:
        print(f"Something went wrong! {e}")
    finally:
        plt.close()

if __name__ == "__main__":
    X, y = load_wine_quality()
    M = 50

    # Split data into a fixed test set and a pool for training samples
    X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sample_sizes = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000,
                    4000, 5000, (X_pool.shape[0] - 1)]

    # Models with fixed parameters
    lasso = Lasso(alpha=0.01)
    knn = KNeighborsRegressor(n_neighbors=10)
    tree = DecisionTreeRegressor(max_depth=5)
    tree_full = DecisionTreeRegressor()  # Fully grown tree
    models = [lasso, knn, tree, tree_full]

    output_dir = "Q2_4"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for model in models:
        sizes, errors, variances, biases_residuals = experiment(
            model, sample_sizes, X_pool, y_pool, X_test, y_test, M
        )

        model_name = model.__class__.__name__
        if model_name == "DecisionTreeRegressor":
            if model.get_params()["max_depth"] is None:
                model_name = "DecisionTreeRegressor_Full"
            else:
                model_name = "DecisionTreeRegressor_Shallow"

        fname = os.path.join(output_dir, f"{model_name}_complexity_curve_sample_size.pdf")
        plot_complexity_curve(fname, sizes, errors, variances, biases_residuals)
