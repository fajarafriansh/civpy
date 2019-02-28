"""
========================
Math (:mod:`civpy.math`)
========================

Contains math classes and functions.


Geometry
========
.. autosummary::
    :toctree: generated/

    Ellipse


Linear Algebra
==============
.. autosummary::
    :toctree: generated/

    projection_angles
    rotation_matrix2
    rotation_matrix3
    rotate2
    rotate3


Optimization
============
.. autosummary::
    :toctree: generated/

    fsolve
"""

from .ellipse import *
from .linalg import *
from .optimize import *
