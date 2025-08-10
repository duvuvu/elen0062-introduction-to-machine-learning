"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from data import make_dataset
from plot import plot_boundary
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil
import os


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
        plt.plot(complexity_values, train_accuracies, label="Train Accuracy")
        plt.plot(complexity_values, test_accuracies, label="Test Accuracy")
        plt.title('')
        plt.xlabel("Max. depth")
        plt.ylabel("Accuracy")
        plt.xticks(complexity_values)
        plt.legend()
        plt.savefig(fname)
    except Exception as e:
        print(f"Something went wrong! {e}")
    finally:
        plt.close()


if __name__ == "__main__":
    classifier = Perceptron  # Do not remove

    n_points = 3000
    test_size = 2/3
    complexity_params = {'max_iter': [5], 'eta0': [1.0], 'tol': [1e-3]}

    seed = 15
    output_dir = classifier().__class__.__name__ + f"_oneGen{seed}"
    metrics = train(classifier, n_points, test_size, complexity_params, seed,
                    output_dir, plot_decision_boundary=True)

    seeds = [5, 6, 7, 8, 9]
    output_dir = ""
    metrics_all, df_summary_metrics = train_n_generations(
        classifier, complexity_params, seeds, n_points, test_size,
        output_dir, plot_decision_boundary=True)