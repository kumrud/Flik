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


"""Limited-memory BFGS (L-BFGS) methods."""


__all__ = [
    "LBfgs",
    ]


class LBfgs:
    """Class for limited-memory BFGS (L-BFGS) methods."""

    def __init__(self, f, J):
        """
        Initialize the L-BFGS methods.

        Parameters
        ----------
        f : callable
        J : callable

        """

        if not callable(f):
            raise TypeError("Argument 'f' must be callable")
        if not callable(J):
            raise TypeError("Argument 'J' must be callable")
        self._function = f
        self._jacobian = J

    def step(self, b, A):
        """
        """
        raise NotImplementedError

    def update(self, new_x, dx, df):
        """
        """
        raise NotImplementedError
