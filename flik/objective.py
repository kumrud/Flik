# An experimental local optimization package
# Copyright (C) 2018 Ayers Lab <ayers@mcmaster.ca>.
#
# This file is part of Flik.
#
# Flik is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Flik is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>


r"""Objective function classes."""


from numbers import Real

import numpy as np


__all__ = [
    "Objective",
    "ForwardDiffObjective",
    "CentralDiffObjective",
    ]


class Objective:
    r"""
    Objective class with analytical evaluation by callable objective function.

    The Objective class and its subclasses are used for evaluating and updating
    objective functions and Jacobians as part of the Newton iterations.

    """

    def __init__(self, f):
        r"""
        Construct an Objective class for a callable analytical objective function.

        Parameters
        ----------
        f : callable, optional

        """
        if not callable(f):
            raise TypeError("f must be a callable object")
        self._function = f

    def __call__(self, x):
        r"""
        Compute the Objective at position ``x``.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        y :np.ndarray

        """
        return self._function(x)


class FiniteDiffObjective(Objective):
    r"""Finite difference Objective class."""

    def __init__(self, f, eps=1.0e-6):
        r"""
        Construct an approximate Objective object from a cost function.

        Parameters
        ----------
        f : callable
        eps : float, optional

        """
        if not callable(f):
            raise TypeError("Argument f must be a callable object")
        if not isinstance(eps, Real):
            raise TypeError("Argument eps must be a real number")
        self._function = f
        self._eps = float(eps)


class ForwardDiffObjective(FiniteDiffObjective):
    r"""Forward difference Objective class."""

    def __call__(self, x):
        r"""
        Compute the Objective at position ``x``.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        y : np.ndarray

        """
        dfx1 = self._function(x)
        f = np.empty_like(x)
        dx = np.copy(x)
        for i, x_i in enumerate(x):
            dx[i] += self._eps
            f[i] = (self._function(dx) - dfx1) / self._eps
            dx[i] = x_i
        return f


class CentralDiffObjective(FiniteDiffObjective):
    r"""Central difference Objective class."""

    def __call__(self, x):
        r"""
        Compute the Objective at position ``x``.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        y : np.ndarray

        """
        f = np.empty_like(x)
        dx = np.copy(x)
        for i, x_i in enumerate(x):
            dx[i] += self._eps
            dfx2 = self._function(dx)
            dx[i] = x_i - self._eps
            dfx1 = self._function(dx)
            f[i] = (dfx2 - dfx1) / (self._eps * 2)
            dx[i] = x_i
        return f
