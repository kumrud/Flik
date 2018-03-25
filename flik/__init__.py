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


"""
An experimental local optimization package.

Copyright (C) 2018 Ayers Lab <ayers@mcmaster.ca>.

"""


# Function classes (objective functions, gradients, Jacobians, Hessians)
from flik.objective import Objective
from flik.objective import ForwardDiffObjective
from flik.objective import CentralDiffObjective
from flik.objective import Objective as Gradient
from flik.objective import ForwardDiffObjective as ForwardDiffGradient
from flik.objective import CentralDiffObjective as CentralDiffGradient
from flik.jacobian import Jacobian
from flik.jacobian import ForwardDiffJacobian
from flik.jacobian import CentralDiffJacobian
from flik.jacobian import Jacobian as Hessian
from flik.jacobian import ForwardDiffJacobian as ForwardDiffHessian
from flik.jacobian import CentralDiffJacobian as CentralDiffHessian

# Line search classes
from flik.line_search import LineSearch
from flik.line_search import ConstantLineSearch
from flik.line_search import QuadraticLineSearch
from flik.line_search import CubicLineSearch
from flik.line_search import BacktraceLineSearch
from flik.line_search import CGLineSearch

# Trust region classes
from flik.trust_region import TrustRegion

# Optimization routines
from flik.optimize import root
from flik.optimize import minimize


__all__ = [
    "Objective",
    "ForwardDiffObjective",
    "CentralDiffObjective",
    "Gradient",
    "ForwardDiffGradient",
    "CentralDiffGradient",
    "Jacobian",
    "ForwardDiffJacobian",
    "CentralDiffJacobian",
    "Hessian",
    "ForwardDiffHessian",
    "CentralDiffHessian",
    "LineSearch",
    "ConstantLineSearch",
    "TrustRegion",
    "root",
    "minimize",
    ]
