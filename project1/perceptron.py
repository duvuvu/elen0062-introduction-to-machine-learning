"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from data import make_dataset
from plot import plot_boundary
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

# (Question 3): Perceptron
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil
import os


class PerceptronClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iter=5, learning_rate=.0001):
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """Fit a perceptron model on (X, y)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float64)
        n_instances, n_features = X.shape

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This class is only dealing with binary "
                             "classification problems")

        # Initialize the vector of trainable parameters (weights)
        self.classes_ = np.unique(y)
        self.weights_ = np.zeros(n_features + 1)  # +1 for the bias term

        # Add bias term (x0) to X
        X_bias = np.hstack([X, np.ones((n_instances, 1))])

        # Training loop over epochs
        for epoch in range(self.n_iter):
            for i in range(n_instances):
                # Linear combination z = w^T x (including bias)
                z = np.dot(X_bias[i], self.weights_)
                # Apply sigmoid activation
                f_pred = self.sigmoid(z)
                gradient = (f_pred - y[i]) * X_bias[i]  # gradient calculation
                # Update weights based on the implementation of SGD
                self.weights_ -= self.learning_rate * gradient

        return self

    def predict(self, X):
        """Predict class for X.
            Each prediction is the index of the maximum probaility value class.
            In case of multiple occurrences of the maximum values, the index
            corresponding to the last occurrence is returned.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """
        y_prob = self.predict_proba(X)
        n_classes = y_prob.shape[1]
        return np.array([n_classes - 1 - np.argmax(y[::-1]) for y in y_prob])

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        X = np.asarray(X, dtype=np.float64)
        n_instances = X.shape[0]
        X_bias = np.hstack([X, np.ones((n_instances, 1))])  # Add bias to X
        z = np.dot(X_bias, self.weights_)  # Compute probabilities
        y_prob = self.sigmoid(z)
        return np.vstack([1 - y_prob, y_prob]).T  # Classes probabilities

    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))


def train(classifier, n_points, test_size, complexity_params,
          index_generation=1, output_dir='', plot_decision_boundary=False):
    classifier_name = classifier().__class__.__name__

    if len(complexity_params) != 1:
        raise ValueError("Dealing with only one complexity parameter!")

    print(f"Run with seed: {index_generation}...")

    # Create folders
    if output_dir == '':
        output_dir = os.path.join(output_dir,
                                  classifier_name,
                                  f"Gen_{index_generation}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    X, y = make_dataset(n_points=n_points, random_state=index_generation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    classifier_params = [(param_name, param_value) for param_name, param_values
                         in complexity_params.items() for param_value
                         in np.asarray(param_values)]

    metrics = {key: [] for key in ["index_generation", classifier_params[0][0],
                                   "train_accuracy", "test_accuracy"]
               }

    for param_name, param_value in classifier_params:
        estimator = classifier(**{param_name: param_value})
        estimator.fit(X_train, y_train)
        if plot_decision_boundary:
            plot_boundary(os.path.join(output_dir, classifier_name + "_bound"
                                       + f"ary_{param_name}_{param_value}"),
                          estimator, X=X_test, y=y_test)

        metrics["index_generation"].append(index_generation)
        metrics[param_name].append(param_value)
        metrics["train_accuracy"].append(estimator.score(X_train, y_train))
        metrics["test_accuracy"].append(estimator.score(X_test, y_test))

    plot_complexity_curve(os.path.join(output_dir,
                          f"{classifier_name}_complexity_curve_accuracy.pdf"),
                          metrics[param_name],
                          metrics["train_accuracy"],
                          metrics["test_accuracy"])
    df_metrics = pd.DataFrame(metrics)
    fname_metrics = os.path.join(output_dir, f"{classifier_name}_metrics.csv")
    df_metrics.to_csv(fname_metrics, index=False, na_rep="NA", sep=";")
    print(df_metrics)
    print("Metrics are stored in file:", fname_metrics)
    print()
    return metrics


def train_n_generations(classifier, complexity_params, generations,
                        n_points=3000, test_size=2/3, output_dir='',
                        plot_decision_boundary=False):
    classifier_name = classifier().__class__.__name__

    if len(complexity_params) != 1:
        raise ValueError("Dealing with only one complexity parameter!")

    generations = np.asarray(generations)
    print(f"Run {len(generations)} generations with seeds: {generations}")

    # Create folders
    if output_dir == '':
        output_dir = os.path.join(output_dir, classifier_name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    classifier_params = [(param_name, param_value) for param_name, param_values
                         in complexity_params.items() for param_value
                         in np.asarray(param_values)]

    param_name = classifier_params[0][0]
    metrics_all = {key: [] for key in ["index_generation", param_name,
                                       "train_accuracy", "test_accuracy"]
                   }

    for seed in generations:
        metrics = train(classifier, n_points, test_size, complexity_params,
                        index_generation=seed,
                        output_dir=os.path.join(output_dir, f"Gen_{seed}"),
                        plot_decision_boundary=plot_decision_boundary)

        metrics_all["index_generation"].extend(metrics["index_generation"])
        metrics_all[param_name].extend(metrics[param_name])
        metrics_all["train_accuracy"].extend(metrics["train_accuracy"])
        metrics_all["test_accuracy"].extend(metrics["test_accuracy"])

    df_all_metrics = pd.DataFrame(metrics_all)
    df_summary_metrics = df_all_metrics \
        .groupby(by=param_name, dropna=False) \
        .agg({"train_accuracy": [np.mean, np.std],
              "test_accuracy": [np.mean, np.std]})
    df_summary_metrics_plot = df_summary_metrics.reset_index()
    df_summary_metrics_plot = df_summary_metrics_plot.dropna(how="any")
    plot_complexity_curve(os.path.join(output_dir, classifier_name
                          + "_complexity_curve_avg_accuracy.pdf"),
                          df_summary_metrics_plot[param_name],
                          df_summary_metrics_plot["train_accuracy"]["mean"],
                          df_summary_metrics_plot["test_accuracy"]["mean"])

    fname_all_metrics = os.path.join(output_dir, classifier_name
                                     + "_metrics_all_generations.csv")
    df_all_metrics.to_csv(fname_all_metrics, index=False, na_rep="NA", sep=";")
    print("All generations metrics are stored in file:", fname_all_metrics)
    print("Summary:", df_summary_metrics, sep="\n")

    return metrics_all, df_summary_metrics


def plot_complexity_curve(fname, complexity_values,
                          train_accuracies, test_accuracies):

    none_indexes = np.where(np.asarray(complexity_values) == None)
    complexity_values = np.delete(complexity_values, none_indexes).tolist()
    train_accuracies = np.delete(train_accuracies, none_indexes).tolist()
    test_accuracies = np.delete(test_accuracies, none_indexes).tolist()

    try:
        plt.figure()
        plt.xscale("log")
        plt.plot(complexity_values, train_accuracies, label="Train Accuracy")
        plt.plot(complexity_values, test_accuracies, label="Test Accuracy")
        plt.title('')
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Accuracy")
        plt.xticks(complexity_values)
        plt.legend()
        plt.savefig(fname)
    except Exception as e:
        print(f"Something went wrong! {e}")
    finally:
        plt.close()


if __name__ == "__main__":
    classifier = PerceptronClassifier  # Do not remove

    n_points = 3000
    test_size = 2/3
    complexity_params = {'learning_rate': [1e-4, 5e-4, 1e-3, 1e-2, 1e-1]}

    seed = 15
    output_dir = classifier().__class__.__name__ + f"_oneGen{seed}"
    metrics = train(classifier, n_points, test_size, complexity_params, seed,
                    output_dir, plot_decision_boundary=True)

    seeds = [5, 6, 7, 8, 9]
    output_dir = ""
    metrics_all, df_summary_metrics = train_n_generations(
        classifier, complexity_params, seeds, n_points, test_size,
        output_dir, plot_decision_boundary=True)
