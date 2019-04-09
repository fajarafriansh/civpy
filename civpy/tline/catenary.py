"""
Copyright (c) 2019, Matt Pewsey
"""

import numpy as np
from math import cos, sin, atan2, ceil
import matplotlib.pyplot as plt
from ..math import fsolve

__all__ = [
    'belly_length',
    'sag',
    'position',
    'stressed_length',
    'average_tension_1',
    'average_tension_2',
    'unstressed_length',
    'max_tension',
    'catenary_const_1',
    'catenary_const_2',
    'catenary_const_3',
    'catenary_const_4',
    'catenary_const_5',
    'plot_catenary_2d',
    'plot_catenary_3d',
    'Catenary',
]


def belly_length(c, l, dz=0):
    """
    Returns the horizontal distance between the back support and belly of
    the sag.

    Parameters
    ----------
    c : float
        The catenary constant.
    l : float
        The span length.
    dz : float
        The elevation change.
    """
    a = c * np.sinh(0.5 * l / c)
    return 0.5 * l - c * np.arcsinh(0.5 * dz / a)


def sag(x, c, l, dz=0):
    """
    Returns the sag in the span.

    Parameters
    ----------
    x : float or array
        The span fraction where the sag will be returned.
    c : float
        The catenary constant.
    l : float
        The span length.
    dz : float
        The elevation change.
    """
    x = np.asarray(x)
    b = belly_length(c, l, dz) / c
    a = x * l / c - b

    return x * dz - c * (np.cosh(a) - np.cosh(b))


def position(x, c, l, dz=0, ba=0):
    """
    Returns the position of the wire measured relative to the back span.

    Parameters
    ----------
    x : float or array
        The span fraction where the sag will be returned.
    c : float
        The catenary constant.
    l : float
        The span length.
    dz : float
        The elevation change.
    ba : float
        The blowout angle of the wire.
    """
    x = np.asarray(x)
    d = sag(x, c, l, dz)

    xp = x * l
    yp = d * sin(ba)
    zp = x * dz - d * cos(ba)

    p = np.column_stack([xp, yp, zp])

    if len(x.shape) == 0:
        return p.ravel()

    return p


def stressed_length(c, l, dz=0):
    """
    Returns the stressed length of the span.

    Parameters
    ----------
    c : float
        The catenary constant.
    l : float
        The span length.
    dz : float
        The elevation change.
    """
    a = 2*c * np.sinh(0.5 * l / c)
    return (a**2 + dz**2)**0.5


def average_tension_1(c, w, l, dz=0):
    """
    Returns the average tension in the wire based on the *catenary constant*.

    Parameters
    ----------
    c : float
        The catenary constant.
    w : float
        The unit load.
    l : float
        The span length.
    dz : float
        The elevation change.
    """
    s = stressed_length(c, l, dz)
    a = belly_length(c, l, dz) / c
    b =  l / c - a

    a = np.sinh(a) * np.cosh(a) + a
    b = np.sinh(b) * np.cosh(b) + b

    return (0.5 * w * c**2 / s) * (a + b)


def average_tension_2(curve, wire, su, c, l, to, tc=None, epo=0, epc=0, dz=0):
    """
    Returns the average tension in the wire based on the *wire strains*.

    Parameters
    ----------
    curve : {'initial', 'creep'}
        The applied loading curve.
    wire : :class:`.Wire`
        The wire.
    su : float
        The unstressed length of the wire.
    c : float
        The catenary constant.
    l : float
        The span length.
    to : float
        The temperature of the wire outer strands.
    tc : float
        The temperature of the wire core strands. If None, and the wire has
        multiple layers, the temperature will be assumed to be the same as
        the outer strands.
    epo : float
        The plastic strain in the outer strands.
    epc : float
        The plastic strain in the core strands.
    dz : float
        The elevation change.
    """
    s = stressed_length(c, l, dz)
    ep = s / su - 1
    return wire.average_tension(curve, ep, to, tc, epo, epc)


def unstressed_length(curve, wire, c, w, l, to, tc=None, epo=0, epc=0, dz=0):
    """
    Returns the unstressed length of the wire.

    Parameters
    ----------
    curve : {'initial', 'creep'}
        The applied loading curve.
    wire : :class:`.Wire`
        The wire.
    c : float
        The catenary constant.
    w : float
        The unit load.
    l : float
        The span length.
    to : float
        The temperature of the wire outer strands.
    tc : float
        The temperature of the wire core strands. If None, and the wire has
        multiple layers, the temperature will be assumed to be the same as
        the outer strands.
    epo : float
        The plastic strain in the outer strands.
    epc : float
        The plastic strain in the core strands.
    dz : float
        The elevation change.
    """
    def func(x):
        b = average_tension_2(curve, wire, x[0], c, l, to, tc, epo, epc, dz)
        return [a - b['force']]

    a = average_tension_1(c, w, l, dz)

    s = (l**2 + dz**2)**0.5
    s, = fsolve(func, [s])

    return s


def max_tension(c, w, l, dz=0):
    """
    Returns the maximum tension in the wire.

    Parameters
    ----------
    c : float
        The catenary constant.
    w : float
        The unit load.
    l : float
        The span length.
    dz : float
        The elevation change.
    """
    b = belly_length(c, l, dz)
    b = np.max([b, l - b], axis=0)
    return c * w * np.cosh(b / c)


def catenary_const_1(h, w):
    """
    Returns the catenary constant based on the *horizontal tension* in the
    wire.

    Parameters
    ----------
    h : float
        The horizontal tension.
    w : float
        The unit load.
    """
    return h / w


def catenary_const_2(s, l, dz=0):
    """
    Returns the catenary constant based on the *stressed length* of the wire.

    Parameters
    ----------
    s : float
        The stressed length.
    l : float
        The span length.
    dz : float
        The elevation change.
    """
    def func(x):
        a = stressed_length(x[0], l, dz)
        return [s - a]

    c = l + 0.5 * dz**2 / l

    if s <= c:
        raise ValueError('Stressed length must be greater than {:.2f}.'.format(c))

    c = (l**3 / (24 * (s - c)))**0.5
    c, = fsolve(func, [c])

    return c


def catenary_const_3(xp, yp, zp, l, dz=0):
    """
    Returns the catenary constant based on a local wire *fit point*.

    Parameters
    ----------
    xp, yp, zp : float
        The local fit point coordinates.
    l : float
        The span length.
    dz : float
        The elevation change.
    """
    def func(x):
        p = position(f, x[0], l, dz, x[1])
        return p[1:] - q

    f = xp / l
    c = abs(0.5 * (xp**2 - xp*l) / zp)
    d = abs(f * dz - zp)
    ba = atan2(yp, d)
    q = np.array([yp, zp])

    if f < 0 or f > 1:
        raise ValueError('Point out of span bounds.')

    if zp > f * dz:
        raise ValueError('Point above span chord.')

    c = fsolve(func, [c, ba])

    return c


def catenary_const_4(tmax, w, l, dz=0):
    """
    Returns the catenary constant based on the *maximum tension* in the wire.

    Parameters
    ----------
    tmax : float
        The maximum tension.
    w : float
        The unit load.
    l : float
        The span length.
    dz : float
        The elevation change.
    """
    def func(x):
        a = max_tension(x[0], w, l, dz)
        return [tmax - a]

    c = tmax / w
    c, = fsolve(func, [c])

    return c


def catenary_const_5(curve, wire, su, w, l, to, tc=None, epo=0, epc=0, dz=0):
    """
    Returns the catenary constant based on the *unstressed length* in the wire.

    Parameters
    ----------
    curve : {'initial', 'creep'}
        The applied loading curve.
    wire : :class:`.Wire`
        The wire.
    su : float
        The unstressed length.
    w : float
        The unit load.
    l : float
        The span length.
    to : float
        The temperature of the wire outer strands.
    tc : float
        The temperature of the wire core strands. If None, and the wire has
        multiple layers, the temperature will be assumed to be the same as
        the outer strands.
    epo : float
        The plastic strain in the outer strands.
    epc : float
        The plastic strain in the core strands.
    dz : float
        The elevation change.
    """
    def func(x):
        s = unstressed_length(curve, wire, x[0], w, l, to, tc, epo, epc, dz)
        return [s - su]

    c = l + 0.5 * dz**2 / l

    if c <= su:
        c = 0.9 * su

    c = (l**3 / (24 * (su - c)))**0.5
    c, = fsolve(func, [c])

    return c


def plot_catenary_2d(c, l, dz=0, ba=0, step=1, ax=None, symbols={}):
    """
    Plots the catenary elevation in 2D.

    Parameters
    ----------
    c : float
        The catenary constant.
    l : float
        The span length.
    dz : float
        The elevation change.
    ba : float
        The blowout angle.
    step : float
        The step interval used for the plot.
    ax
        The axes to which the plot will be added. If None, a new figure and
        axes will be created.
    symbols : dict
        A dictionary of plot symbols.
    """
    n = int(ceil(l / step))
    x = np.linspace(0, 1, n)
    x = position(x, c, l, dz, ba)

    if ax is None:
        mx = x.max(axis=0)
        c = 0.5 * (mx + x.min(axis=0))
        r = 1.1 * np.max(mx - c)
        xlim, _, ylim = np.column_stack([c - r, c + r])

        fig = plt.figure()
        ax = fig.add_subplot(111,
            xlabel='X',
            ylabel='Y',
            xlim=xlim,
            ylim=ylim,
            aspect='equal'
        )
        ax.grid()

    sym = dict(
        curve='b-',
        points=None
    )
    sym.update(symbols)

    if sym['curve'] is not None:
        ax.plot(x[:,0], x[:,2], sym['curve'])

    if sym['points'] is not None:
        ax.plot(x[:,0], x[:,2], sym['points'])

    return ax


def plot_catenary_3d(c, l, dz=0, ba=0, step=1, ax=None, symbols={}):
    """
    Plots the catenary elevation in 2D.

    Parameters
    ----------
    c : float
        The catenary constant.
    l : float
        The span length.
    dz : float
        The elevation change.
    ba : float
        The blowout angle.
    step : float
        The step interval used for the plot.
    ax
        The axes to which the plot will be added. If None, a new figure and
        axes will be created.
    symbols : dict
        A dictionary of plot symbols.
    """
    n = int(ceil(l / step))
    x = np.linspace(0, 1, n)
    x = position(x, c, l, dz, ba)

    if ax is None:
        mx = x.max(axis=0)
        c = 0.5 * (mx + x.min(axis=0))
        r = 1.1 * np.max(mx - c)
        xlim, ylim, zlim = np.column_stack([c - r, c + r])

        fig = plt.figure()
        ax = fig.add_subplot(111,
            projection='3d',
            xlabel='X',
            ylabel='Y',
            zlabel='Z',
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            aspect='equal'
        )
        ax.grid()

    sym = dict(
        curve='b-',
        points=None
    )
    sym.update(symbols)

    if sym['curve'] is not None:
        ax.plot(x[:,0], x[:,1], x[:,2], sym['curve'])

    if sym['points'] is not None:
        ax.plot(x[:,0], x[:,1], x[:,2], sym['points'])

    return ax


class Catenary(object):
    def __init__(self, inode, jnode, wire):
        self.inode = inode
        self.jnode = jnode
        self.wire = wire
        self.reset()

    def reset(self):
        self._unstr_length = None
        self._outer_plastic_strain = 0
        self._core_plastic_strain = 0
        self._controlling_constraint = None
        self._controlling_load = None
        self._tolerances = {}

    def get_tolerances(self):
        tol = dict(
            unstr_length=0,
            outer_plastic_strain=0,
            core_plastic_strain=0,
        )

        tol.update(self._tolerances)
        return tol

    def span(self):
        return np.linalg.norm(self.jnode[:2] - self.inode[:2])

    def elev_change(self):
        return self.jnode.z - self.inode.z

    def azimuth(self):
        dx, dy = self.jnode[:2] - self.inode[:2]
        return atan2(dx, dy)

    def horz_unit_load(self, load_case):
        weather = load_case.weather
        ri = weather.ice_thickness
        pw = weather.wind_pressure()

        dw = self.wire.diameter
        waz = load_case.wind_azimuth
        wh = pw * (dw + 2*ri)

        if waz is None:
            return wh

        return wh * cos(self.azimuth() - waz)**2

    def vert_unit_load(self, load_case):
        weather = load_case.weather
        ri = weather.ice_thickness
        rhoi = weather.ice_density

        wire = self.wire
        dw = wire.diameter
        w = wire.unit_weight

        return rhoi * np.pi * ri * (dw + ri) + w

    def unit_load(self, load_case):
        wh = self.horz_unit_load(load_case)
        wv = self.vert_unit_load(load_case)
        return (wh**2 + wv**2)**0.5 + load_case.weather.k_factor

    def blowout_angle(self, load_case):
        wh = self.horz_unit_load(load_case)
        wv = self.vert_unit_load(load_case)
        ang = atan2(wh, wv)
        s = np.sign(ang - self.azimuth())

        return s * ang

    def find_unstr_length_1(self, tmax, load_case):
        if tmax <= 1:
            tmax = tmax * self.wire.breaking_strength

        w = self.unit_load(load_case)
        l = self.span()
        dz = self.elev_change()

        curve = load_case.condition
        weather = load_case.weather
        to = weather.wire_temperature
        tc = weather.wire_core_temperature
        epo = epc = 0

        c = catenary_const_4(tmax, w, l, dz)
        s = unstressed_length(curve, self.wire, c, w, l, to, tc, epo, epc, dz)

        if self._unstr_length is None or s > self._unstr_length:
            self._unstr_length = s
            self._controlling_constraint = load_case

    def find_unstr_length_2(self, xp, yp, zp, load_case):
        x = np.array([xp, yp, zp]) - self.inode

        ux = self.jnode[:2] - self.inode[:2]
        ux = ux / np.linalg.norm(ux)
        uy = np.array([ux[1], -ux[0]])

        xp = ux.dot(x[:2])
        yp = uy.dot(x[:2])
        zp = x[2]

        w = self.unit_load(load_case)
        l = self.span()
        dz = self.elev_change()

        curve = load_case.condition
        weather = load_case.weather
        to = weather.wire_temperature
        tc = weather.wire_core_temperature
        epo = epc = 0

        c, _ = catenary_const_3(xp, yp, zp, l, dz)
        s = unstressed_length(curve, self.wire, c, w, l, to, tc, epo, epc, dz)

        if self._unstr_length is None or s > self._unstr_length:
            self._unstr_length = s
            self._controlling_constraint = load_case

    def find_plastic_strains(self, load_case):
        w = self.unit_load(load_case)
        l = self.span()
        dz = self.elev_change()

        weather = load_case.weather
        to = weather.wire_temperature
        tc = weather.wire_core_temperature
        su = self._unstr_length
        epo = self._outer_plastic_strain
        epc = self._core_plastic_strain
        curve = load_case.wire_condition

        c = catenary_const_5(curve, wire, su, w, l, to, tc, epo, epc, dz)
        odict = average_tension_2(curve, wire, su, c, l, to, tc, epo, epc, dz)

        if odict['core'] is not None and odict['core']['plastic_strain'] > epc:
            self._core_plastic_strain = odict['core']['plastic_strain']

        if odict['outer']['plastic_strain'] > epo:
            self._outer_plastic_strain = odict['outer']['plastic_strain']
            self._controlling_load = load_case

    def find_sag_tension(self, load_case):
        w = self.unit_load(load_case)
        l = self.span()
        dz = self.elev_change()

        weather = load_case.weather
        to = weather.wire_temperature
        tc = weather.wire_core_temperature
        su = self._unstr_length
        epo = epc = 0

        c = catenary_const_5('initial', wire, su, w, l, to, tc, epo, epc, dz)
        h = c * w
        d = sag(0.5, c, l, dz)

        odict = dict(
            initial_catenary_const=c,
            initial_horz_tension=h,
            initial_sag=d,
        )

        epo = self._outer_plastic_strain
        epc = self._core_plastic_strain

        c = catenary_const_5('initial', wire, su, w, l, to, tc, epo, epc, dz)
        h = c * w
        d = sag(0.5, c, l, dz)

        odict.update(
            final_catenary_const=c,
            final_horz_tension=h,
            final_sag=d,
        )

        return odict

    def find_unstr_length_tol_1(self):
        self._tolerances['unstr_length'] = s

    def find_unstr_length_tol_2(self):
        self._tolerances['unstr_length'] = s

    def find_plastic_strain_tol(self):
        self._tolerances['core_plastic_strain'] = epc
        self._tolerances['outer_plastic_strain'] = epo
