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


r"""Jacobian classes."""


from numbers import Integral
from numbers import Real

import numpy as np


__all__ = [
    "Jacobian",
    "ForwardDiffJacobian",
    "CentralDiffJacobian",
    ]


class Jacobian:
    r"""
    Jacobian class with analytical evaluation by callable Jacobian function.

    The Jacobian class and its subclasses are used for evaluating and updating
    Jacobians as part of the Newton iterations.

    """

    def __init__(self, jac):
        r"""
        Construct a Jacobian class for a callable analytical jacobian.

        Parameters
        ----------
        jac : callable, optional

        Raises
        ------
        TypeError
            If an argument of an invalid type or shape is passed.

        """
        if not callable(jac):
            raise TypeError("J must be a callable object")
        self._jac = jac

    def __call__(self, x):
        r"""
        Compute the Jacobian at position ``x``.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        y :np.ndarray

        """
        return self._jac(x)

    @staticmethod
    def step_linear(b, A):
        r"""
        Compute the Newton step ``dx = -1 * inv(J(f(x))) * f(x)``.

        Parameters
        ----------
        b : np.ndarray
            1-dimensional array of the negative of the function evaluated at the
            current guess of the roots -f(x_0).
        A : np.ndarray
            2-dimensional array of the Jacobian evaluated at the current guess of
            the roots J(x_0).

        Returns
        -------
        dx : np.ndarray

        """
        return np.linalg.solve(A, b)

    @staticmethod
    def step_inverse(b, A):
        r"""
        Compute the Newton step ``dx = -1 * inv(J(f(x))) * f(x)``.

        Parameters
        ----------
        b : np.ndarray
            1-dimensional array of the negative of the function evaluated at the
            current guess of the roots -f(x_0).
        A : np.ndarray
            2-dimensional array of the inverse Jacobian evaluated at the current
            guess of the roots J(x_0).

        Returns
        -------
        dx : np.ndarray

        """
        return np.dot(A, b)

    @staticmethod
    def step_gaussnewton(b, A):
        r"""
        Compute the Gauss-Newton step ``dx = -1 * pinv(J(f(x))) * f(x)``.

        Parameters
        ----------
        b : np.ndarray
            1-dimensional array of the negative of the function evaluated at the
            current guess of the roots -f(x_0).
        A : np.ndarray
            2-dimensional array of the Jacobian evaluated at the current guess of
            the roots J(x_0).

        Returns
        -------
        dx : np.ndarray

        """
        return np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))

    def update_newton(self, A, new_x, *_):
        r"""
        Update the Jacobian matrix ``A`` at new solution vector ``x_(k+1)``.

        Parameters
        ----------
        new_x : np.ndarray
            ``x_(k+1)``
        dx : np.ndarray
            ``x_(k+1) - x_k``
        df : np.ndarray
            ``f_(k+1) - f_k``

        """
        A[...] = self(new_x)

    @staticmethod
    def update_goodbroyden(A, _, dx, df):
        r"""
        Update the Jacobian matrix ``A`` at new solution vector ``x_(k+1)``.

        Parameters
        ----------
        new_x : np.ndarray
            ``x_(k+1)``
        dx : np.ndarray
            ``x_(k+1) - x_k``
        df : np.ndarray
            ``f_(k+1) - f_k``

        """
        # Compute Good Broyden right hand side second term numerator
        t = df
        t -= np.dot(A, dx)
        # Divide by dx norm
        t /= np.dot(dx, dx)
        # Compute matrix from dot product of f and transposed dx
        A += np.outer(t, dx.T)

    @staticmethod
    def update_badbroyden(A, _, dx, df):
        r"""
        Update the inverse Jacobian matrix ``A`` at new solution vector ``x_(k+1)``.

        Parameters
        ----------
        new_x : np.ndarray
            ``x_(k+1)``
        dx : np.ndarray
            ``x_(k+1) - x_k``
        df : np.ndarray
            ``f_(k+1) - f_k``

        """
        t2 = np.dot(dx.T, A)
        t1 = dx
        t1 -= np.dot(A, df)
        t1 /= np.dot(t2, df)
        A += np.outer(t1, t2)

    @staticmethod
    def update_dfp(A, _, dx, df):
        r"""
        Update the Jacobian matrix ``A`` at new solution vector ``x_(k+1)``.

        Parameters
        ----------
        new_x : np.ndarray
            ``x_(k+1)``
        dx : np.ndarray
            ``x_(k+1) - x_k``
        df : np.ndarray
            ``f_(k+1) - f_k``

        """
        norm = np.dot(df, dx)
        t1 = np.outer(df, dx.T)
        t1 /= -norm
        t1 += np.eye(t1.shape[0])
        t2 = np.outer(dx, df.T)
        t2 /= -norm
        t2 += np.eye(t2.shape[0])
        A[...] = np.dot(t1, np.dot(A, t2))
        t1 = np.outer(df, df.T)
        t1 /= norm
        A += t1

    @staticmethod
    def update_dfpinv(A, _, dx, df):
        r"""
        Update the inverse Jacobian matrix ``A`` at new solution vector ``x_(k+1)``.

        Parameters
        ----------
        new_x : np.ndarray
            ``x_(k+1)``
        dx : np.ndarray
            ``x_(k+1) - x_k``
        df : np.ndarray
            ``f_(k+1) - f_k``

        """
        t1 = np.outer(dx, dx.T)
        t1 /= np.dot(dx, df)
        A += t1
        t2 = np.dot(A, df)
        t1 = np.dot(t2, t2.T)
        t1 /= np.dot(df, t2)
        A -= t1

    @staticmethod
    def update_bfgs(A, _, dx, df):
        r"""
        Update the Jacobian matrix ``A`` at new solution vector ``x_(k+1)``.

        Parameters
        ----------
        new_x : np.ndarray
            ``x_(k+1)``
        dx : np.ndarray
            ``x_(k+1) - x_k``
        df : np.ndarray
            ``f_(k+1) - f_k``

        """
        Jacobian.update_dfpinv(A, None, df, dx)

    @staticmethod
    def update_bfgsinv(A, _, dx, df):
        r"""
        Update the inverse Jacobian matrix ``A`` at new solution vector ``x_(k+1)``.

        Parameters
        ----------
        new_x : np.ndarray
            ``x_(k+1)``
        dx : np.ndarray
            ``x_(k+1) - x_k``
        df : np.ndarray
            ``f_(k+1) - f_k``

        """
        Jacobian.update_dfp(A, None, df, dx)

    @staticmethod
    def update_sr1(A, _, dx, df):
        r"""
        Update the Jacobian matrix ``A`` at new solution vector ``x_(k+1)``.

        Parameters
        ----------
        new_x : np.ndarray
            ``x_(k+1)``
        dx : np.ndarray
            ``x_(k+1) - x_k``
        df : np.ndarray
            ``f_(k+1) - f_k``

        """
        t1 = df
        t1 -= np.dot(A, dx)
        t2 = np.outer(t1, t1.T)
        t2 /= np.dot(t1.T, dx)
        A += t2

    @staticmethod
    def update_sr1inv(A, _, dx, df):
        r"""
        Update the inverse Jacobian matrix ``A`` at new solution vector ``x_(k+1)``.

        Parameters
        ----------
        new_x : np.ndarray
            ``x_(k+1)``
        dx : np.ndarray
            ``x_(k+1) - x_k``
        df : np.ndarray
            ``f_(k+1) - f_k``

        """
        Jacobian.update_sr1(A, None, df, dx)


class FiniteDiffJacobian(Jacobian):
    r"""Finite difference Jacobian approximation class."""

    def __init__(self, f, m, n=None, eps=1.0e-4):
        r"""
        Construct a finite difference approximate Jacobian function.

        Parameters
        ----------
        f : callable
            The function for which the Jacobian is being approximated.
        m : int
            Size of the function output vector.
        n : int, optional
            Size of the function argument vector (default is ``n`` == ``m``).
        eps : float or np.ndarray, optional
            Increment in the function's argument to use when approximating the
            Jacobian.

        """
        # Check input types and values
        if n is None:
            n = m
        if not callable(f):
            raise TypeError("f must be a callable object")
        if not isinstance(m, Integral):
            raise TypeError("m must be an integral type")
        if not isinstance(n, Integral):
            raise TypeError("n must be an integral type")
        if m <= 0:
            raise ValueError("m must be > 0")
        if n <= 0:
            raise ValueError("n must be > 0")
        if not (isinstance(eps, np.ndarray) and eps.ndim == 1):
            if not isinstance(eps, Real):
                raise TypeError("eps must be a float or 1-dimensional array")
        if isinstance(eps, np.ndarray):
            if eps.size != n:
                raise ValueError("eps must be of the same length as the input vector")
            eps = np.copy(eps)
        else:
            eps = np.full(int(n), float(eps), dtype=np.float)
        if np.any(eps <= 0.0):
            raise ValueError("eps must be > 0.0")
        # Assign internal attributes
        self._function = f
        self._m = int(m)
        self._n = int(n)
        self._eps = eps


class ForwardDiffJacobian(FiniteDiffJacobian):
    r"""Forward difference Jacobian approximation class."""

    def __call__(self, x, fx=None):
        r"""
        Evaluate the approximate Jacobian at position ``x``.

        Parameters
        ----------
        x : np.ndarray
            Argument vector to the approximate Jacobian function.
        fx : np.ndarray, optional
            Output vector of the function at position `x` (optional, but avoids
            an extra function call).

        Returns
        -------
        jacobian : np.ndarray
            Value of the approximate Jacobian at position ``x``.

        """
        # Note: In order to stick to row-major iteration, this algorithm
        # computes the transpose of the approximate Jacobian into the jac
        # vector. This function, being the Jacobian proper, returns the
        # transpose of the jac vector.
        jac = np.empty((self._n, self._m), dtype=np.float)
        # Evaluate function at x (fx = f(x)) if required
        if fx is None:
            fx = self._function(x)
        # Copy x to vector dx
        dx = np.copy(x)
        # Iterate over elements of `x` to increment
        for i, eps_i in enumerate(self._eps):
            # Add forward-epsilon increment to dx (dx = x + e_i * eps_i)
            dx[i] += eps_i
            # Evaluate function at dx (dfx = f(dx))
            dfx = self._function(dx)
            # Calculate df[j]/dx[i] = (dfx - fx) / eps_i into dfx vector
            dfx -= fx
            dfx /= eps_i
            # Put result from dfx into the ith row of the jac matrix
            jac[i, :] = dfx
            # Reset dx = x
            dx[i] = x[i]
        # df[i]/dx[j] = transpose(jac)
        return jac.transpose()


class CentralDiffJacobian(FiniteDiffJacobian):
    r"""Central difference Jacobian approximation class."""

    def __call__(self, x):
        r"""
        Evaluate the approximate Jacobian at position ``x``.

        Parameters
        ----------
        x : np.ndarray
            Argument vector to the approximate Jacobian function.

        Returns
        -------
        jacobian : np.ndarray
            Value of the approximate Jacobian at position ``x``.

        """
        # Note: In order to stick to row-major iteration, this algorithm
        # computes the transpose of the approximate Jacobian into the jac
        # vector. This function, being the Jacobian proper, returns the
        # transpose of the jac vector.
        jac = np.empty((self._n, self._m), dtype=np.float)
        # Copy x to vector dx
        dx = np.copy(x)
        # Iterate over elements of `x` to increment
        for i, (x_i, eps_i) in enumerate(zip(x, self._eps)):
            # Add forward-epsilon increment to dx (+dx = x + e_i * eps_i)
            dx[i] += eps_i
            # Evaluate function at +dx (dfx2 = f(+dx))
            dfx2 = self._function(dx)
            # Add backward-epsilon increment to dx (-dx = x - e_i * eps_i)
            dx[i] = x_i - eps_i
            # Evaluate function at -dx (dfx1 = f(-dx))
            dfx1 = self._function(dx)
            # Calculate df[j]/dx[i] = (dfx2 - dfx1) / (2 * eps_i) into dfx2 vector
            dfx2 -= dfx1
            dfx2 /= 2 * eps_i
            # Put result from dfx2 into the ith row of the jac matrix
            jac[i, :] = dfx2
            # Reset dx = x
            dx[i] = x_i
        # df[i]/dx[j] = transpose(jac)
        return jac.transpose()
