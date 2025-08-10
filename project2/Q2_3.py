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

def experiment(model, param_name, param_values, X_pool, y_pool, X_test, y_test, N=250, M=50):
    expected_errors = []
    model_variances = []
    biases_plus_residuals = []

    for param in param_values:
        setattr(model, param_name, param)
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

    return param_values, expected_errors, model_variances, biases_plus_residuals

def plot_complexity_curve(fname, param_name, complexity_params,
                          expected_error, model_variance,
                          residual_bias, logscale=False):
    try:
        plt.figure(figsize=(5, 5))
        plt.plot(complexity_params, expected_error, label="Expected Error",
                 marker='o',markersize=5, alpha=0.8)
        plt.plot(complexity_params, residual_bias, label="Bias + Residual Error",
                 marker='o', markersize=5, alpha=0.8)
        plt.plot(complexity_params, model_variance, label="Variance", marker='o',
                 markersize=5, alpha=0.8)
        plt.xlabel(param_name)
        if logscale:
            plt.xscale('log')
        plt.ylabel("Error")
        plt.xticks(complexity_params)
        plt.legend()
        plt.savefig(fname)
    except Exception as e:
        print(f"Something went wrong! {e}")
    finally:
        plt.close()

if __name__ == "__main__":
    X, y = load_wine_quality() # Load the dataset
    N = 250
    M = 50

    # Split data into a fixed test set and a pool for training samples
    X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    methods = [dict(model=KNeighborsRegressor,
                    complexity_params=('n_neighbors', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                       12, 14, 16, 18 ,20, 25])
                    ),
                dict(model=Lasso,
                    complexity_params=('alpha', [1.e-03, 1.e-02, 5.e-02, 8.e-02, 1.e-01,
                                                 2.e-01, 3.e-01, 4.e-01, 5.e-01, 5.e-01,
                                                 1.e+00, 1.e+01, 1.e+02])
                    ),
                dict(model=DecisionTreeRegressor,
                    complexity_params=('max_depth', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                     12, 14, 16, 18 ,20, 25])
                    ),
                ]

    output_dir = "Q2_3"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for method in methods:
        model = method['model']()
        complexity_name, complexity_values = method['complexity_params']
        params, errors, variances, biases_residuals = experiment(
            model, complexity_name, complexity_values, X_pool, y_pool, X_test, y_test, N, M
        )

        model_name = model.__class__.__name__
        fname = os.path.join(output_dir, f"{model_name}_complexity_curve_{complexity_name}.pdf")
        if complexity_name == 'alpha':
            plot_complexity_curve(fname, complexity_name,
                                  params, errors, variances,
                                  biases_residuals, logscale=True)
        else:
            plot_complexity_curve(fname, complexity_name,
                                  params, errors, variances,
                                  biases_residuals)
