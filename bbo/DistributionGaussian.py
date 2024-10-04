# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
#
# DmpBbo is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# DmpBbo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
""" Module for the DistributionGaussian class. """


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def plot_error_ellipse(mu, cov, ax=None, **kwargs):
    """
    Plot the error ellipse at a point given its covariance matrix

    Parameters
    ----------
    mu : array (2,)
    The center of the ellipse

    cov : array (2,2)
    The covariance matrix for the point

    ax : matplotlib.Axes, optional
    The axis to plot on

    **kwargs : dict
    These keywords are passed to matplotlib.patches.Ellipse

    From https://github.com/dfm/dfmplot/blob/master/dfmplot/ellipse.py
    """

    facecolor = kwargs.pop("facecolor", "none")
    edgecolor = kwargs.pop("edgecolor", "k")

    x, y = mu
    U, S, V = np.linalg.svd(cov)  # noqa
    theta = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    ellipse_plot = Ellipse(
        xy=(x, y),
        width=2 * np.sqrt(S[0]),
        height=2 * np.sqrt(S[1]),
        angle=float(theta),
        facecolor=facecolor,
        edgecolor=edgecolor,
        **kwargs,
    )

    if not ax:
        ax = plt.gca()
    lines = ax.add_patch(ellipse_plot)
    return lines, ax


class DistributionGaussian:
    """ A class for representing a Gaussian distribution.
    """

    def __init__(self, mean=0.0, covar=1.0):
        """ Construct the Gaussian distribution with a mean and covariance matrix.

        @param mean: Mean of the distribution
        @param covar: Covariance matrix of the distribution
        """
        self.mean = mean
        self.covar = covar
        self.if_clip = False
        self.upper_limit = np.nan
        self.lower_limit = np.nan
        self.dim = covar.shape[0]
        
    def set_range(self, upper_limit, lower_limit):
        """
        set range limitation, to be used in the self.clip()
        Args:
            upper_limit (1-D list, of length N): upper_limit of samples
            lower_limit (1-D list, of length N): lower_limit of samples
        """
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.if_clip = True
        
    def clip(self, x:np.ndarray):
        """
        clip x according to the self.upper_limit & self.lower_limit.
        Args:
            samples (np.array): (size n_samples X dim(mean)
        """
        
        temp = np.zeros(x.shape)
        for i in range(x.shape[1]):
            np.clip(x[:,i], self.lower_limit[i], self.upper_limit[i], temp[:,i])

        # print(temp)
        return temp

    def generate_samples(self, n_samples=1):
        """ Generate samples from the distribution & clip them

        @param n_samples: Number of samples to sample
        @return: The samples themselves (size n_samples X dim(mean)
        """
        samples = np.random.multivariate_normal(self.mean, self.covar, n_samples)
        
        if self.if_clip:
            samples = self.clip(samples)            
        return samples

    def max_eigen_value(self):
        """ Get the largest eigenvalue of the covariance matrix of this distribution.

        @return: largest eigenvalue of the covariance matrix of this distribution.
        """
        return max(np.linalg.eigvals(self.covar))

    def __str__(self):
        """ Get a string representation of an object of this class.

        @return: A string representation of an object of this class.
        """
        return f"N( {self.mean}, {self.covar})"

    def plot(self, ax=None):
        """ Plot the Gaussian distribution.

        @param ax: Axis to plot on (default: None => create a new axis)
        @return: The axis on which plotting took place.
        """
        if not ax:
            ax = plt.axes()
        return plot_error_ellipse(self.mean[:2], self.covar[:2, :2], ax)
