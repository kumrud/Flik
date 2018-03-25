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

from flik.objective import Objective
from flik.objective import CentralDiffObjective

from flik.jacobian import Jacobian
from flik.jacobian import CentralDiffJacobian

from flik.line_search import LineSearch
from flik.line_search import ConstantLineSearch

from flik.trust_region import TrustRegion

from flik.methods.lbfgs import LBfgs


__all__ = [
    "root",
    "minimize",
    ]


def root(f, x, J=None, eps=1.0e-6, maxiter=100,
    method="newton", linesearch=None, trustregion=None, **kwargs):
    r"""
    Solve a system of nonlinear equations with the Newton method.

    Parameters
    ----------
    f : callable
        Vector-valued function corresponding to nonlinear system of equations.
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
    linesearch : (float | np.ndarray | dict | LineSearch), optional
        Scaling factor for Newton step (if float or np.ndarray).
        Configuration options (if dict).
        Otherwise the LineSearch object is passed directly.
    trustregion : (dict | LineSearch), optional
        Configuration options (if dict).
        Otherwise the TrustRegion object is passed directly.
    kwargs : dict
        Options specific to certain methods.

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
    # Check optimization parameter types
    if not (isinstance(x, np.ndarray) and x.ndim == 1):
        raise TypeError("Argument x should be a 1-dimensional numpy array")
    if not isinstance(eps, Real):
        raise TypeError("Argument eps should be a real number")
    if not isinstance(maxiter, Integral):
        raise TypeError("Argument maxiter should be an integer number")
    # Check f (objective function)
    if callable(f):
        f = f if isinstance(f, Objective) else Objective(f)
    else:
        raise TypeError("Argument f should be callable")
    # Check J (Jacobian function) types
    if J is None:
        J = CentralDiffJacobian(f, f(x).shape[0], x.shape[0])
    elif callable(J):
        J = J if isinstance(J, Jacobian) else Jacobian(J)
    else:
        raise TypeError("Argument J should be callable")
    # Check optimization parameter values
    if eps < 0.0:
        raise ValueError("Argument eps should be >= 0.0")
    if maxiter < 1:
        raise ValueError("Argument maxiter should be >= 1")
    # Check method, linesearch, and trustregion arguments,
    # and choose the step/update/linesearch/trustregion functions
    setup = _root_setup(f, J, method, linesearch, trustregion, kwargs)
    # Return result of Newton iterations
    return _root_iteration(f, x, J, float(eps), int(maxiter), setup)


def minimize(f, x, grad=None, H=None, eps=1.0e-6, maxiter=100,
    method="newton", linesearch=None, trustregion=None, **kwargs):
    r"""
    Minimize a multivariate, nonlinear function with the Newton method.

    Parameters
    ----------
    f : callable
        Function to be minimized.
    x : np.ndarray
    grad : callable or Gradient, optional
    H : callable or Hessian, optional
    eps : float, optional
        Convergence threshold for vector function grad norm.
    maxiter : int, optional
        Maximum number of iterations to perform.
    method : str, optional
        Update method for the (approximated) H(x) or the inverse of H(x). The
        default uses Newton method.
    linesearch : (float | np.ndarray | LineSearch), optional
        Scaling factor for Newton step (if float or np.ndarray).
        Otherwise the LineSearch must be specified.
    kwargs : dict
        Options specific to certain methods.

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
            Function value at solution.
        grad
            Gradient evaluated at solution.
        H
            Hessian evaluated at solution.
        eps
            Convergence threshold for vector function f norm.

    """
    # Check grad type
    if grad is None:
        grad = CentralDiffObjective(f)
    # Minimization problem is equivalent to root-finding with
    # objective function == grad and Hessian = Jacobian(grad).
    result = root(grad, x, H, eps, maxiter,
                  method, linesearch, trustregion, **kwargs)
    # Rename keys for ``minimize`` output
    result["grad"] = result.pop("f")
    result["H"] = result.pop("J")
    # Compute final function value
    result["f"] = f(result["x"])
    # Return output dictionary
    return result


def _root_iteration(f, x, J, eps, maxiter, setup):
    r"""Run the iterations for ``root``."""
    # Unpack the step/update/linesearch/trustregion functions
    inverse, direct, step, update, linesearch, trustregion = setup
    # Compute b = f(x_0)
    b = f(x)
    # Compute A = J(x_0) or A = J^-1(x_0) if inverse flag is True
    # and direct flag is False
    if direct:
        A = None
    elif inverse:
        A = np.linalg.inv(J(x))
    else:
        A = J(x)
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
            linesearch(dx, x)
            trustregion(dx, x)
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


def _root_setup(f, J, method, linesearch, trustregion, kwargs):
    r"""Return the functions used in ``root`` according to the method."""
    # Check method type
    if not isinstance(method, str):
        raise TypeError("Argument 'method' must be a string")
    method = method.lower()
    # Get inverse flag
    inverse = method.endswith("inv") or method in ("badbroyden",)
    # Get direct flag
    direct = method in ("lbfgs",)
    # Get step function for simple methods
    if inverse:
        step = J.step_inverse
    elif method == "gaussnewton":
        step = J.step_gaussnewton
    else:
        step = J.step_linear
    # Get update function for simple methods
    if method in ("newton", "gaussnewton"):
        update = J.update_newton
    elif method in ("broyden", "goodbroyden"):
        update = J.update_goodbroyden
    elif method in ("broydeninv", "badbroyden"):
        update = J.update_badbroyden
    elif method == "dfp":
        update = J.update_dfp
    elif method == "dfpinv":
        update = J.update_dfpinv
    elif method == "bfgs":
        update = J.update_bfgs
    elif method == "bfgsinv":
        update = J.update_bfgsinv
    elif method == "sr1":
        update = J.update_sr1
    elif method == "sr1inv":
        update = J.update_sr1inv
    # Get step and update functions for direct methods
    elif method == "lbfgs":
        lbfgs = LBfgs(f, J, **kwargs)
        step = lbfgs.step
        update = lbfgs.update
    # Catch invalid method option
    else:
        raise ValueError("Argument method is not a valid option")
    # Check linesearch type
    if linesearch is None:
        linesearch = LineSearch(f, J)
    elif isinstance(linesearch, dict):
        linesearch = LineSearch.fromdict(f, J, linesearch)
    elif isinstance(linesearch, (Real, np.ndarray)):
        linesearch = ConstantLineSearch(f, J, linesearch)
    elif not isinstance(linesearch, LineSearch):
        raise TypeError("Argument linesearch should be a "
                        "float, numpy array, dict, or LineSearch object")
    # Check trustregion type
    if trustregion is None:
        trustregion = TrustRegion(f, J)
    elif isinstance(trustregion, dict):
        trustregion = TrustRegion.fromdict(f, J, trustregion)
    elif not isinstance(trustregion, TrustRegion):
        raise TypeError("Argument linesearch should be a "
                        "float, numpy array, dict, or TrustRegion object")
    return inverse, direct, step, update, linesearch, trustregion
