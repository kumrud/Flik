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


r"""Classes for trust regions."""


__all__ = [
    "TrustRegion",
    ]


class TrustRegion:
    r"""Base trust region class."""

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
    def fromdict(kwargs):
        r"""
        Return a TrustRegion object from a configuration dict.

        Parameters
        ----------
        kwargs : dict

        Returns
        -------
        trustregion : TrustRegion

        """
        if not isinstance(kwargs, dict):
            raise TypeError("Argument kwargs must be a dict")
        method = kwargs.pop("method").lower()
        if method == "dogleg":
            raise NotImplementedError
        elif method == "doubledogleg":
            raise NotImplementedError
        elif method == "cauchy":
            raise NotImplementedError
        elif method == "steinhaug":
            raise NotImplementedError
        else:
            raise ValueError("Invalid 'method' argument")
