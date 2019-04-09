"""
Copyright (c) 2019, Matt Pewsey
"""

import copy
import propy
import numpy as np
import scipy.optimize
from math import cos, sin, atan2, ceil
from shapely.geometry import Polygon
from .linalg import rotate2

__all__ = ['Ellipse']


class Ellipse(np.ndarray):
    """
    A class representing an ellipse.

    Parameters
    ----------
    x, y : float
        The x, y coordinates for the center of the ellipse.
    width : float
        The width of the ellipse in the local x direction.
    height : float
        The height of the ellipse in the local y direction.
    rotation : float
        The counter clockwise rotation of the ellipse.
    """
    x = propy.index_property(0)
    y = propy.index_property(1)

    def __new__(cls, x, y, width, height, rotation=0):
        obj = np.array([x, y], dtype='float').view(cls)
        obj.width = width
        obj.height = height
        obj.rotation = rotation
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.width = getattr(obj, 'width', 0)
        self.height = getattr(obj, 'height', 0)
        self.rotation = getattr(obj, 'rotation', 0)

    def __repr__(self):
        s = ('x', 'y', 'width', 'height', 'rotation')
        s = ('{}={!r}'.format(k, getattr(self, k)) for k in s)
        return '{}({})'.format(type(self).__name__, ', '.join(s))

    def copy(self):
        return copy.copy(self)

    def point(self, angle):
        """
        Returns the (x, y) point on the perimeter of the ellipse located at
        the specified local angle.

        Parameters
        ----------
        angle : float
            The local angle, measured from the ellipse width axis, for which
            the point will be returned.
        """
        a = 0.5 * self.width
        b = 0.5 * self.height
        p = np.array([a * cos(angle), b * sin(angle)])
        x0 = np.asarray(self)

        return rotate2(p, self.rotation) + x0

    def points(self, step=0.1):
        """
        Returns points around the boundary of the ellipse.

        Parameters
        ----------
        step : float
            The step distance used for polygon generation.
        """
        a = 0.5 * self.width
        b = 0.5 * self.height

        p = self.perimeter()
        n = int(ceil(p / step))
        ang = np.linspace(0, 2*np.pi, n)
        p = np.column_stack([a * np.cos(ang), b * np.sin(ang)])
        x0 = np.asarray(self)

        return rotate2(p, self.rotation) + x0

    def contains(self, point):
        """
        Returns true if the input point is contained within the ellipse.

        Parameters
        ----------
        point : array
            The (x, y) point for which the check will be performed.
        """
        p = np.asarray(point) - self
        x, y = rotate2(p, -self.rotation)

        return (x / self.width)**2 + (y / self.height)**2 <= 0.25

    def perimeter(self):
        """
        Returns the perimeter of the ellipse.
        """
        a = 0.5 * self.width
        b = 0.5 * self.height

        h = ((a - b) / (a + b))**2
        c = 1 + h/4 + h**2/64 + h**3/256 + 25*h**4/16384 + 49*h**5/65536

        return np.pi * (a + b) * c

    def polygon(self, step=0.1):
        """
        Returns an inscribed polygon for the perimeter of the ellipse.

        Parameters
        ----------
        step : float
            The step distance used for polygon generation.
        """
        p = self.points(step)
        return Polygon(p)

    def area(self):
        """
        Returns the area of the ellipse.
        """
        return 0.25 * np.pi * self.width * self.height

    def lap_area(self, ellipse, step=1e-4):
        """
        Returns the area of overlap with the input ellipse.

        Parameters
        ----------
        ellipse : :class:`.Ellipse`
            The ellipse to calculate the distance to.
        step : float
            The step distance used for polygon generation.
        """
        p = self.polygon(step)
        q = ellipse.polygon(step)

        return p.intersection(q).area

    def lap_fractions(self, ellipse, step=1e-4):
        """
        Returns the overlap area fractions between the ellipses.

        Parameters
        ----------
        ellipse : :class:`.Ellipse`
            The ellipse to calculate the distance to.
        step : float
            The step distance used for polygon generation.
        """
        lap = self.lap_area(ellipse, step)
        a = lap / self.area()
        b = lap / ellipse.area()
        return a, b

    def distance(self, ellipse):
        """
        Calculates the minimum distance between two ellipses. If the ellipses
        overlap, the distance is negative and indicative of the maximum
        overlap.

        Parameters
        ----------
        ellipse : :class:`.Ellipse`
            The ellipse to calculate the distance to.

        Returns
        -------
        distance : float
            The minimum distance between the ellipses.
        points : array
            An array of points of shape (2, 2) defining the minimum
            distance line.
        """
        def func(x):
            p1 = self.point(x[0])
            p2 = ellipse.point(x[1])
            s = -1 if self.contains(p2) or ellipse.contains(p1) else 1
            return s * np.linalg.norm(p1 - p2)

        dx, dy = ellipse - self
        a1 = atan2(dx, dy) - self.rotation
        a2 = atan2(-dx, -dy) - ellipse.rotation
        r = scipy.optimize.minimize(func, [a1, a2], method='BFGS')

        if not r.success:
            raise ValueError('Method failed with final values:\n{}'.format(r))

        p = np.array([self.point(r.x[0]), ellipse.point(r.x[1])])

        return r.fun, p
