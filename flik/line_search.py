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

    def __init__(self, f, J):
        r"""Initialize the object."""
        if not callable(f):
            raise ValueError("f must be callable")
        if not callable(J):
            raise ValueError("J must be callable")
        self._function = f
        self._jacobian = J

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
    def fromdict(f, J, kwargs):
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
        method = kwargs.pop("method").lower()
        if method == "constant":
            return ConstantLineSearch(f, J, **kwargs)
        elif method == "quadratic":
            return ConstantLineSearch(f, J, **kwargs)
        elif method == "cubic":
            return ConstantLineSearch(f, J, **kwargs)
        elif method == "backtrace":
            return BacktraceLineSearch(f, J, **kwargs)
        elif method == "cg":
            return CGLineSearch(f, J, **kwargs)
        else:
            raise ValueError("Invalide 'method' argument.")


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
        ConstantLineSearch.__init__(self, f, J, constant=constant)

    def __call__(self, dx, x):
        r"""
        """
        raise NotImplementedError


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
