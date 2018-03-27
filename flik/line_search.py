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
    "QuadraticLineSearch",
    "CubicLineSearch",
    "BacktraceLineSearch",
    "CGLineSearch",
    ]


class LineSearch:
    r"""Base line search class."""

    def __init__(self, f, J, x, dx, a):
        r"""Initialize the object."""
        if not callable(f):
            raise ValueError("f must be callable")
        if not callable(J):
            raise ValueError("J must be callable")
        self._function = f
        self._jacobian = J
        self._x = x
        self._dx = dx
        self._a = a

    def __call__(self, *_):
        r"""
        Apply the line search to the function ``f`` at position vector
        ``x`` and direction vector ``dx``.

        Parameters
        ----------
        dx : np.ndarray
        x : np.ndarray
        f : callable

        """
        pass

    @staticmethod
    def from_dict(f, J, x, dx, a, kwargs):
        r"""
        Return a LineSearch object from a configuration dict.

        Parameters
        ----------
        kwargs : dict

        Returns
        -------
        linesearch : LineSearch

        """
        if not isinstance(kwargs, dict):
            raise TypeError("Argument kwargs must be a dict")
        
        # get conditions from dict
        if 'condition' in kwargs.keys():
            cond = kwargs.pop("condition").lower()

        method = kwargs.pop("method").lower()
        if method == "constant":
            return ConstantLineSearch(f, J, x, dx, a, cond)
        elif method == "quadratic":
            return QuadraticLineSearch(f, J, x, dx, a, cond)
        elif method == "cubic":
            return CubicLineSearch(f, J, x, dx, a, cond)
        elif method == "backtrace":
            return BacktraceLineSearch(f, J, x, dx, a, cond)
        elif method == "cg":
            return CGLineSearch(f, J, x, dx, a, cond)
        else:
            raise ValueError("Invalid 'method' argument.")

    @staticmethod
    def satify_conditions(f, J, x, dx, a, cond):
        """
        Check if line algorithm conditions are satisfied.
        
        Parameters
        ----------
        cl : list
            list with condition names
        """
        satisfy = []
        for condition in cond:
            if condition == "soft-wolfe":
                satisfy.append(soft_wolfe(f, J, x, dx, a))
            if condition == "strong-wolfe":
                satisfy.append(strong_wolfe(f, J, x, dx, a))
            if condition == "armijo":
                satisfy.append(armijo(f, J, x, dx, a))
            else:
                raise ValueError("Invalid 'condition' argument.")
        return all(satify)

    @staticmethod
    def soft_wolfe(J, x, dx, a):
        pass


class ConstantLineSearch(LineSearch):
    r"""Basic line search that scales the direction vector by a constant."""

    def __init__(self, f, J, constant=0.5):
        r"""
        Initialize the ConstantLineSearch with a scalar or vector constant.

        Parameters
        ----------
        constant : float or np.ndarray

        """
        LineSearch.__init__(self, f, J)
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

        """
        dx *= self._constant


class QuadraticLineSearch(ConstantLineSearch):
    r"""
    """

    def __init__(self, f, J, constant=0.5):
        r"""
        """
        ConstantLineSearch.__init__(self, f, J, x, dx, a)

    def __call__(self):
        r"""
        """
        while True:
            # interpolate until conditions satisfied
	        pz = f(x)
            pzp = np.dot(J(x), dx)
            pa = f(np.outer(x +  guess * dx))
            a = - pzp * guess**2
            a /= 2 * (pa - pz - pzp * guess)
            if self.satisfy_conditions(f, J, x, dx, a):
                dx *= a


class CubicLineSearch(ConstantLineSearch):
    r"""
    """

    def __init__(self, f, J, constant=0.5):
        r"""
        """
        ConstantLineSearch.__init__(self, f, J, constant=constant)

    def __call__(self, dx, x):
        r"""
        """
        raise NotImplementedError


class BacktraceLineSearch(ConstantLineSearch):
    r"""
    """

    def __init__(self, f, J, constant=0.5):
        """
        """
        ConstantLineSearch.__init__(self, f, J, constant=constant)

    def __call__(self, dx, x):
        r"""
        """
        raise NotImplementedError


class CGLineSearch(LineSearch):
    r"""
    """

    def __init__(self, f, J, **kwargs):
        """
        """
        LineSearch.__init__(self, f, J)

    def __call__(self, dx, x):
        r"""
        """
        raise NotImplementedError
