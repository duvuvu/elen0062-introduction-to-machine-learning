# (Question 4): Method Comparison
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from data import make_dataset
from perceptron import PerceptronClassifier
import matplotlib.pyplot as plt
import os
import csv


def tune_hyperparameters(classifier, param_grid, X_train, y_train,
                         cv=5, scoring='accuracy'):
    grid_search = GridSearchCV(classifier(), param_grid,
                               cv=cv, scoring=scoring)
    grid_search.fit(X_train, y_train)

    # Get the best model, parameters, and score
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params


def train(classifier, complexity_params, n_points, test_size,
          index_generation=1, k_fold=5, scoring='accuracy',
          n_irrelevant_features=0):
    X, y = make_dataset(n_points=n_points, n_irrelevant=n_irrelevant_features,
                        random_state=index_generation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    param_name, param_values = complexity_params
    param_grid = {param_name: param_values}
    best_model, best_params = tune_hyperparameters(classifier, param_grid,
                                                   X_train, y_train,
                                                   k_fold, scoring)

    # Extract key and value from best_params
    for key, value in best_params.items():
        param_name = key
        param_value = value

    estimator = classifier(**best_params)
    estimator.fit(X_train, y_train)

    # Calculate metrics
    train_accuracy = estimator.score(X_train, y_train)
    test_accuracy = estimator.score(X_test, y_test)

    metric = {
        'index_generation': index_generation,
        'classifier': classifier().__class__.__name__,
        'param_name': param_name,
        'param_value': param_value,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }

    return metric


def train_1_generation(classifiers, n_points, test_size, seed, k_fold=5,
                       scoring='accuracy', n_irrelevant_features=0):
    metrics = {key: [] for key in ['index_generation', 'classifier',
                                   'param_name', 'param_value',
                                   'train_accuracy', 'test_accuracy']
               }

    for key, value in classifiers.items():
        classifier = value[0]
        complexity_params = value[1]

        metric = train(classifier, complexity_params, n_points, test_size,
                       seed, k_fold, scoring, n_irrelevant_features)
        metrics["index_generation"].append(metric['index_generation'])
        metrics["classifier"].append(metric['classifier'])
        metrics["param_name"].append(metric['param_name'])
        metrics["param_value"].append(metric['param_value'])
        metrics["train_accuracy"].append(metric['train_accuracy'])
        metrics["test_accuracy"].append(metric['test_accuracy'])

    return metrics


def train_n_generations(classifiers, n_points, test_size, seeds, k_fold=5,
                        scoring='accuracy', n_irrelevant_features=0,
                        output_dir=''):
    # Create folders
    if output_dir == '':
        output_dir = os.path.join(output_dir, "MethodComparison")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    metrics_all = {key: [] for key in ['index_generation', 'classifier',
                                       'param_name', 'param_value',
                                       'train_accuracy', 'test_accuracy']
                   }

    for seed in seeds:
        metrics = train_1_generation(classifiers, n_points, test_size, seed,
                                     k_fold, scoring, n_irrelevant_features)

        metrics_all["index_generation"].extend(metrics["index_generation"])
        metrics_all["classifier"].extend(metrics["classifier"])
        metrics_all["param_name"].extend(metrics["param_name"])
        metrics_all["param_value"].extend(metrics["param_value"])
        metrics_all["train_accuracy"].extend(metrics["train_accuracy"])
        metrics_all["test_accuracy"].extend(metrics["test_accuracy"])

    save_metrics_to_csv(metrics_all, os.path.join(output_dir,
                        "MethodComparison_metrics_all_irrelevant"
                        + f"_{n_irrelevant_features}.csv"))

    metrics_avg = {key: [] for key in ['classifier', 'train_accuracy_mean',
                                       'train_accuracy_std',
                                       'test_accuracy_mean',
                                       'test_accuracy_std']}

    classifier_lst = []
    for key, value in classifiers.items():
        classifier = value[0]().__class__.__name__
        classifier_lst.append(classifier)

        train_accuracies = [metrics_all["train_accuracy"][i] for i
                            in range(len(metrics_all["classifier"]))
                            if metrics_all["classifier"][i] == classifier]
        test_accuracies = [metrics_all["test_accuracy"][i] for i
                           in range(len(metrics_all["classifier"]))
                           if metrics_all["classifier"][i] == classifier]

        avg_train_accuracy = round(np.mean(train_accuracies), 4)
        avg_test_accuracy = round(np.mean(test_accuracies), 4)
        std_train_accuracy = round(np.std(train_accuracies, ddof=1), 4)
        std_test_accuracy = round(np.std(test_accuracies, ddof=1), 4)

        metrics_avg["classifier"].append(classifier)
        metrics_avg["train_accuracy_mean"].append(avg_train_accuracy)
        metrics_avg["train_accuracy_std"].append(std_train_accuracy)
        metrics_avg["test_accuracy_mean"].append(avg_test_accuracy)
        metrics_avg["test_accuracy_std"].append(std_test_accuracy)

    save_metrics_to_csv(metrics_avg, os.path.join(output_dir,
                        "MethodComparison_metrics_avg_irrelevant"
                         + f"_{n_irrelevant_features}.csv"))

    return metrics_avg


def save_metrics_to_csv(metrics_dict, file_path):
    # Get the headers from the dictionary keys
    headers = metrics_dict.keys()

    # Transpose the dictionary values to rows
    rows = zip(*metrics_dict.values())

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write the headers
        writer.writerows(rows)    # Write the rows


def plot_accuracy_comparison(fname, classifiers, accuracy_irrelevant_0,
                             accuracy_irrelevant_200, title=''):
    x = np.arange(len(classifiers))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, accuracy_irrelevant_0, width,
                   label='n_irrelevant=0')
    bars2 = ax.bar(x + width/2, accuracy_irrelevant_200, width,
                   label='n_irrelevant=200')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Classifier')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers)
    ax.legend()

    # Set y-axis range
    ax.set_ylim(0.7, 1)

    # Attach a text label above each bar in *bars*, displaying its height.
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(round(height, 4)),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)
    plt.savefig(fname)


if __name__ == "__main__":
    n_points = 3000
    test_size = 2/3
    output_dir = ''
    seeds = [5, 6, 7, 8, 9]
    k_fold = 10
    scoring = 'accuracy'
    classifiers = {
        'DecisionTree': (DecisionTreeClassifier,
                         ('max_depth', [1, 2, 4, 8, None])),
        'KNeighbors': (KNeighborsClassifier,
                       ('n_neighbors', [1, 5, 50, 100, 500])),
        'Perceptron': (PerceptronClassifier,
                       ("learning_rate", [1e-4, 5e-4, 1e-3, 1e-2, 1e-1]))
    }

    # Question 2
    n_irrelevant_features = 0
    metrics_avg_irrelevant_0 = train_n_generations(
                                classifiers, n_points, test_size, seeds,
                                k_fold, scoring, n_irrelevant_features,
                                output_dir)
    print(metrics_avg_irrelevant_0)

    # Question 3
    n_irrelevant_features = 200
    metrics_avg_irrelevant_200 = train_n_generations(
                                  classifiers, n_points, test_size, seeds,
                                  k_fold, scoring, n_irrelevant_features,
                                  output_dir)
    print(metrics_avg_irrelevant_200)

    # Question 4
    if output_dir == '':
        output_dir = os.path.join(output_dir, "MethodComparison")
    plot_accuracy_comparison(os.path.join(output_dir,
                             "MethodComparison_train_accuracy_comparison.pdf"
                                          ),
                             metrics_avg_irrelevant_0["classifier"],
                             metrics_avg_irrelevant_0["train_accuracy_mean"],
                             metrics_avg_irrelevant_200["train_accuracy_mean"],
                             title='')
    plot_accuracy_comparison(os.path.join(output_dir,
                             "MethodComparison_test_accuracy_comparison.pdf"),
                             metrics_avg_irrelevant_0["classifier"],
                             metrics_avg_irrelevant_0["test_accuracy_mean"],
                             metrics_avg_irrelevant_200["test_accuracy_mean"],
                             title='')
