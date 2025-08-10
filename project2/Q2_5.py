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
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from data import load_wine_quality

def experiment(ensemble_model, base_model, param_name, param_values, X_pool, y_pool, X_test, y_test, N=250, M=50):
    expected_errors = []
    variances = []
    biases_plus_residuals = []

    for param in param_values:
        setattr(ensemble_model, param_name, param)
        ensemble_model.estimator = base_model
        predictions = []

        for _ in range(M):
            X_train, _, y_train, _ = train_test_split(X_pool, y_pool, train_size=N,
                                                      random_state=None)
            ensemble_model.fit(X_train, y_train)
            y_pred = ensemble_model.predict(X_test)
            predictions.append(y_pred)

        predictions = np.array(predictions)

        expected_error = np.mean((predictions - y_test) ** 2)
        expected_errors.append(expected_error)

        variance = np.mean(np.var(predictions, axis=0))
        variances.append(variance)

        bias_plus_residual = expected_error - variance
        biases_plus_residuals.append(bias_plus_residual)

    return param_values, expected_errors, variances, biases_plus_residuals

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
        plt.xlabel("n_estimators")
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
    n_estimators = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200]
    N = 250
    M = 50

    # Split data into a pool for training samples and a fixed test set
    X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    shallow_tree = DecisionTreeRegressor(max_depth=5)
    full_tree = DecisionTreeRegressor()
    estimators = [shallow_tree, full_tree]

    output_dir = "Q2_5"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for estimator in estimators:
        # Bagging
        bagging = BaggingRegressor(estimator=estimator, random_state=42)
        bagging_params, bagging_errors, bagging_variances, bagging_biases = experiment(
            bagging, estimator, 'n_estimators', n_estimators, X_pool, y_pool, X_test, y_test, N, M)

        # Boosting: GradientBoostingRegressor 
        boosting = GradientBoostingRegressor(n_estimators=1, random_state=42)
        boosting_params, boosting_errors, boosting_variances, boosting_biases = experiment(
            boosting, estimator, 'n_estimators', n_estimators, X_pool, y_pool, X_test, y_test, N, M)

        if estimator.get_params()["max_depth"] is None:
            estimator_name = "DecisionTreeRegressor_Full"
        else:
            estimator_name = "DecisionTreeRegressor_Shallow"
        fname = os.path.join(output_dir, f"{estimator_name}_complexity_curve_bagging.pdf")
        plot_complexity_curve(fname, bagging_params, bagging_errors, bagging_variances, bagging_biases)

        fname = os.path.join(output_dir, f"{estimator_name}_complexity_curve_boosting.pdf")
        plot_complexity_curve(fname, boosting_params, boosting_errors, boosting_variances, boosting_biases)
