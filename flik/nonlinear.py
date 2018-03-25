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


r"""
Solvers for nonlinear systems using the Newton and Gauss-Newton algorithms.

Functions used to find the roots of a nonlinear system of equations given the
residual function `f` and its analytical Jacobian `J`. An exactly-determined
system (`m` equations, `m` variables) is best solved with the Newton method.
Over- or under- determined systems (`m` equations, `n` variables) must be
solved in the least-squares sense using the Gauss-Newton method.

"""


from numbers import Integral
from numbers import Real

import numpy as np

from flik.jacobian import Jacobian

from flik.approx_jacobian import CentralDiffJacobian

from flik.line_search import LineSearch
from flik.line_search import ConstantLineSearch

from flik.trust_region import TrustRegion


__all__ = [
    "nonlinear_solve",
    ]


def nonlinear_solve(f, x, J=None, eps=1.0e-6, maxiter=100,
    method="newton", linesearch=None, trustregion=None):
    r"""
    Solve a system of nonlinear equations with the Newton method.

    Parameters
    ----------
    f : callable
        Vector-valued function corresponding to nonlinear system of equations.
        Must be of the form f(x), where x is a 1-dimensional array.
    x : np.ndarray
        Solution initial guess.
    J : callable or Jacobian, optional
        Jacobian of function f. Must be of the form J(x), where x is a
        1-dimensional array. If none is given, then the Jacobian is calculated
        using finite differences.
    eps : float, optional
        Convergence threshold for vector function f norm.
    maxiter : int, optional
        Maximum number of iterations to perform.
    method : str, optional
        Update method for the (approximated) J(x) or the inverse of J(x). The
        default uses Newton method.
    linesearch : (float | np.ndarray | LineSearch), optional
        Scaling factor for Newton step (if float or np.ndarray).
        Otherwise the LineSearch must be specified.

    Returns
    -------
    result : dict
        A dictionary with the keys:
        success
            Boolean variable informing whether the algorithm succeeded or not.
        message
            Information about the cause of the termination.
        niter
            Number of actual iterations performed.
        x
            Nonlinear system of equations solution (Root).
        f
            Vector function evaluated at solution.
        J
            Jacobian evaluated at solution.
        eps
            Convergence threshold for vector function f norm.

    """
    # Check input types
    if not callable(f):
        raise TypeError("Argument f should be callable")
    # Check J (Jacobian function) type
    if J is None:
        J = CentralDiffJacobian(f, f(x).shape[0], x.shape[0])
    elif callable(J) and not isinstance(J, Jacobian):
        J = Jacobian(J)
    else:
        raise TypeError("Argument J should be callable or a Jacobian object")
    # Check optimization parameter types
    if not (isinstance(x, np.ndarray) and x.ndim == 1):
        raise TypeError("Argument x should be a 1-dimensional numpy array")
    if not isinstance(eps, Real):
        raise TypeError("Argument eps should be a real number")
    if not isinstance(maxiter, Integral):
        raise TypeError("Argument maxiter should be an integer number")
    # Check optimization parameter values
    if eps < 0.0:
        raise ValueError("Argument eps should be >= 0.0")
    if maxiter < 1:
        raise ValueError("Argument maxiter should be >= 1")
    eps = float(eps)
    maxiter = int(maxiter)
    # Check method, linesearch, and trustregion arguments,
    # and choose the step/update/linesearch/trustregion functions
    setup = _nonlinear_setup(f, J, method, linesearch, trustregion)
    # Return result of Newton iterations
    return _nonlinear_iteration(f, x, J, eps, maxiter, setup)


def _nonlinear_iteration(f, x, J, eps, maxiter, setup):
    r"""Run the iterations for ``newton_solve``."""
    # Unpack the step/update/linesearch/trustregion functions
    inverse, step, update, linesearch, trustregion = setup
    # Compute b = f(x_0)
    b = f(x)
    # Compute A = J(x_0) or A = J^-1(x_0) if inverse flag is True
    A = np.linalg.inv(J(x)) if inverse else J(x)
    # Run Newton iterations
    success = False
    message = "Maximum number of iterations reached."
    for niter in range(1, maxiter + 1):
        # Compute b = -f(x_k)
        b *= -1
        # Attempt to find a suitable Newton step
        try:
            # Calculate step direction dx
            dx = step(b, A)
            # Apply line search and trust region to step direction
            linesearch(dx, x, f)
            trustregion(dx, x, f)
        # Check if step, linesearch, or trustregion throw LinAlgError
        except np.linalg.LinAlgError:
            # If so, we're done (fail)
            A = None
            b *= -1
            # TODO: change message to something more general (and in tests)
            message = "Singular Jacobian; no solution found."
            break
        # Apply Newton step to x_k
        x += dx
        # Compute df = f(x_(k+1)) - f(x_k) and b = f(x_(k+1))
        df = b
        b = f(x)
        df += b
        # Compute A = J(x_(k+1)) or A = J^-1(x_(k+1)) (in-place update)
        update(A, x, dx, df)
        # Check for convergence of f(x_(k+1)) to zero
        if np.linalg.norm(b) < eps:
            # If so, we're done (success)
            success = True
            message = "Convergence obtained."
            break
    # Compute A = inv(A) if inverse flag is True
    if inverse and A is not None:
        A = np.linalg.inv(A)
    # Return result dictionary
    return {
        "success": success,
        "message": message,
        "niter": niter,
        "x": x,
        "f": b,
        "J": A,
        "eps": eps,
        }


def _nonlinear_setup(f, J, method, linesearch, trustregion):
    r"""Return the functions used ``newton_solve`` according to the method."""
    # Check method type
    if not isinstance(method, str):
        raise TypeError("Argument 'method' must be a string")
    method = method.lower()
    # Get inverse flag
    inverse = method.endswith("inv")
    # Get step function
    if inverse:
        step = _step_inverse
    elif method == "gaussnewton":
        step = _step_gaussnewton
    else:
        step = _step_linear
    # Get update function
    if method in ("newton", "gaussnewton"):
        update = J.update_newton
    elif method == "broyden":
        update = J.update_goodbroyden
    elif method == "broydeninv":
        update = J.update_badbroyden
    elif method == "dfp":
        update = J.update_dfp
    elif method == "bfgsinv":
        update = J.update_bfgsinv
    elif method == "sr1":
        update = J.update_sr1
    elif method == "sr1inv":
        update = J.update_sr1inv
    else:
        raise ValueError("Argument method is not a valid option")
    # Check linesearch type
    if linesearch is None:
        linesearch = LineSearch()
    elif isinstance(linesearch, (Real, np.ndarray)):
        linesearch = ConstantLineSearch(linesearch)
    elif not isinstance(linesearch, LineSearch):
        raise TypeError("Argument linesearch should be a "
                        "float, numpy array, or LineSearch object")
    # Check trustregion type
    if trustregion is None:
        trustregion = TrustRegion()
    elif not isinstance(trustregion, TrustRegion):
        raise TypeError("Argument linesearch should be a "
                        "float, numpy array, or LineSearch object")
    return inverse, step, update, linesearch, trustregion


def _step_linear(b, A):
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


def _step_inverse(b, A):
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


def _step_gaussnewton(b, A):
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
