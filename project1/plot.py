"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

def make_cmaps():
    """
    Return
    ------
    bg_map, sc_map: tuple (colormap, colormap)
        bg_map: The colormap for the background
        sc_map: Binary colormap for scatter points
    """
    top = mpl.cm.get_cmap('Oranges_r')
    bottom = mpl.cm.get_cmap('Blues')

    newcolors = np.vstack((top(np.linspace(.25, 1., 128)),
                           bottom(np.linspace(0., .75, 128))))
    bg_map = ListedColormap(newcolors, name='OrangeBlue')

    sc_map = ListedColormap(['#ff8000', 'DodgerBlue'])

    return bg_map, sc_map


def plot_boundary(fname, fitted_estimator, X, y, mesh_step_size=0.1, title=""):
    """Plot estimator decision boundary and scatter points

    Parameters
    ----------
    fname : str
        File name where the figures is saved.

    fitted_estimator : a fitted estimator

    X : array, shape (n_samples, 2)
        Input matrix

    y : array, shape (n_samples, )
        Binary classification target

    mesh_step_size : float, optional (default=0.2)
        Mesh size of the decision boundary

    title : str, optional (default="")
        Title of the graph

    """
    bg_map, sc_map = make_cmaps()

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))

    Z = fitted_estimator.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    try:
        plt.title(title)
        plt.xlabel('$X_1$')
        plt.ylabel('$X_2$')

        # Put the result into a color plot
        cf = plt.contourf(xx, yy, Z, cmap=bg_map, alpha=0.8, vmin=0, vmax=1)

        # Plot testing point
        plt.scatter(X[::5, 0], X[::5, 1], c=y[::5], cmap=sc_map, edgecolor='black',
                    s=10)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        # Plot true function
        xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
        lims = [
            [np.min(xlim), np.max(ylim)],
            [np.max(xlim), np.min(ylim)]
        ]
        plt.axline([0, -np.log(3)/(3*0.391)], [-np.log(3)/(3*0.391), 0], c='k')

        plt.clim(np.min(Z), np.max(Z))
        plt.colorbar(cf)
        plt.savefig('{}.pdf'.format(fname))
    finally:
        plt.close()
