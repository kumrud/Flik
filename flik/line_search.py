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


r"""Classes for line searches."""


from numbers import Real

import numpy as np


__all__ = [
    "LineSearch",
    "ConstantLineSearch",
    ]


class LineSearch:
    r"""Base line search class."""

    def __init__(self, a):
        r"""Initialize the object."""
        self._a = a

    def __call__(self, dx):
        r"""
        Apply the line search to the function ``f`` at position vector
        ``x`` and direction vector ``dx``.

        Parameters
        ----------
        dx : np.ndarray
        x : np.ndarray
        f : callable
        a : float, int, np.ndarray
            inital guess for alpha

        """
        dx *= self._a

    def quad_inter(self, f, g, x, dx):
        """
        Quadratic interpolation
        """
        pz = f(x)
        pzp = np.dot(g(x), dx)
        pa = f(np.outer(x + self._a * dx))
        n = - pzp * self._a**2
        n /= 2 * (pa - pz - pzp * self._a)
        self._a = n

    @staticmethod
    def soft_wolfe_cond(a, g, x, dx, c2=None):
        pap = g(np.outer(x, a*dx))
        pzp = g(x)
        left = np.dot(dx, pap)
        right = c2 * np.dot(dx, pzp)
        return left >= right


class ConstantLineSearch(LineSearch):
    r"""Basic line search that scales the direction vector by a constant."""

    def __init__(self, constant):
        r"""
        Initialize the ConstantLineSearch with a scalar or vector constant.

        Parameters
        ----------
        constant : float or np.ndarray

        """
        if isinstance(constant, Real):
            constant = float(constant)
        elif not (isinstance(constant, np.ndarray) and constant.ndim == 1):
            raise TypeError("Argument constant should be a "
                            "float or 1-dimensional numpy array")
        if np.any(constant < 0.0):
            raise ValueError("Argument constant should be >= 0.0")
        self._constant = constant

    def __call__(self, dx, *_):
        r"""
        Apply the line search to the function ``f`` at position vector
        ``x`` and direction vector ``dx``.

        Parameters
        ----------
        dx : np.ndarray
        x : np.ndarray
        f : callable

        """
        dx *= self._constant
