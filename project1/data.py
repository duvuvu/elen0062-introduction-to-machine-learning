"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import check_random_state
from plot import make_cmaps


def make_dataset(n_points, class_prop=.25, std=1.6, n_irrelevant=0, random_state=None):
    """Generate a binary classification dataset of two circular gaussians.

    Parameters
    ----------
    n_points: int (>0)
        Number of data points.
    class_prop: float (0 < class_prop < 1, default=.5)
        The proportion of positive class instances.
    std: float (>0, default: 1.6)
        The standard deviation of the gaussians.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    X : array of shape [n_points, 2]
        The input samples.

    y : array of shape [n_points]
        The output values.
    """
    drawer = check_random_state(random_state)

    n_pos = int(n_points*class_prop)

    y = np.zeros((n_points), dtype=int)
    X = drawer.normal((1.5, 1.5), scale=std, size=(n_points, 2))

    X[:n_pos] *= -1
    y[:n_pos] = 1

    _X = drawer.normal(size=(n_points, n_irrelevant), scale=std)
    X = np.concatenate([X, _X], axis=1)

    shuffler = np.arange(n_points)
    drawer.shuffle(shuffler)

    return X[shuffler], y[shuffler]


if __name__ == '__main__':

    X, y = make_dataset(1000)
    LABELS = ['Negative', 'Positive']
    print('Number of positive examples:', np.sum(y))
    print('Number of negative examples:', np.sum(y==0))

    _, sc_map = make_cmaps()
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    ax = plt.axes()
    ax.set_axisbelow(True)
    plt.grid(True)

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=sc_map, edgecolor='black',
                    s=10)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')

    handles, labels = scatter.legend_elements()
    for ha in handles:
        ha.set_mec('black')
    legend = ax.legend(handles, LABELS, loc='upper left')

    plt.gca().set_aspect('equal')
    plt.show()

