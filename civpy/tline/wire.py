import propy
import numpy as np
import matplotlib.pyplot as plt

from ..math import poly
from ..engr import mechanics
from ..data import query_wire

__all__ = ['Wire']


class Wire(object):
    # Custom properties
    name = propy.str_property('name')
    resistance_coeffs = propy.array_property('resistance_coeffs')
    core_initial_coeffs = propy.array_property('core_initial_coeffs')
    core_creep_coeffs = propy.array_property('core_creep_coeffs')
    outer_initial_coeffs = propy.array_property('outer_initial_coeffs')
    outer_creep_coeffs = propy.array_property('outer_creep_coeffs')

    def __init__(self, name, area, unit_weight, diameter,
                 # Thermal calculations
                 core_diameter=None,
                 radial_conductivity=None,
                 solar_emissivity=0.5,
                 solar_absorptivity=0.5,
                 resistance_coeffs=[],
                 # Stress strain
                 ref_temperature=None,
                 linear_strain=0.005,
                 compression_limit=0,
                 breaking_strength=None,
                 # Core strands
                 core_elasticity=None,
                 core_thermal_coeff=None,
                 core_initial_coeffs=[],
                 core_creep_coeffs=[],
                 # Outer strands
                 outer_elasticity=None,
                 outer_thermal_coeff=None,
                 outer_initial_coeffs=[],
                 outer_creep_coeffs=[],
                 # Propagation error analysis
                 tolerances={},
                 **kwargs
        ):
        self.name = name
        self.area = area
        self.unit_weight = unit_weight
        self.diameter = diameter
        self.meta = dict(**kwargs)

        # Thermal calculations
        self.core_diameter = core_diameter
        self.radial_conductivity = radial_conductivity
        self.solar_emissivity = solar_emissivity
        self.solar_absorptivity = solar_absorptivity
        self.resistance_coeffs = resistance_coeffs
        self.temperature_data = None
        self.resistance_data = None

        # Stress strain
        self.ref_temperature = ref_temperature
        self.linear_strain = linear_strain
        self.compression_limit = compression_limit
        self.breaking_strength = breaking_strength

        # Core strands
        self.core_elasticity = core_elasticity
        self.core_thermal_coeff = core_thermal_coeff
        self.core_initial_coeffs = core_initial_coeffs
        self.core_creep_coeffs = core_creep_coeffs

        # Outer strands
        self.outer_elasticity = outer_elasticity
        self.outer_thermal_coeff = outer_thermal_coeff
        self.outer_initial_coeffs = outer_initial_coeffs
        self.outer_creep_coeffs = outer_creep_coeffs

        # Propagation error analysis
        self.tolerances = tolerances

    def __repr__(self):
        s = ('name',)
        s = ('{}={!r}'.format(k, getattr(self, k)) for k in s)
        return '{}({})'.format(type(self).__name__, ', '.join(s))

    @classmethod
    def from_db(cls, name, include_meta=False, **kwargs):
        odict = query_wire(name)
        odict.update(kwargs)

        # Resistance coeffs
        deg = odict.pop('resistance_deg', None)
        temp = odict.pop('temperature_data', None)
        resist = odict.pop('resistance_data', None)

        # Create object
        obj = cls(**odict).from_pct_strain()

        if not include_meta:
            obj.meta.clear()

        if deg is not None and temp is not None and resist is not None:
            obj.fit_resistances(temp, resist, deg)

        return obj

    def get_tolerances(self):
        tol = dict(
            area=0,
            unit_weight=0,
            diameter=0,
            core_diameter=0,
            radial_conductivity=0,
            solar_emissivity=0,
            solar_absorptivity=0,
            ref_temperature=0,
            linear_strain=0,
            compression_limit=0,
            core_elasticity=0,
            core_thermal_coeff=0,
            outer_elasticity=0,
            outer_thermal_coeff=0,
        )

        tol.update(self.tolerances)

        return tol

    def fit_resistances(self, temp, resist, deg=1):
        self.resistance_coeffs = poly.polyfit(temp, resist, deg)
        self.temperature_data = temp
        self.resistance_data = resist

        return self.resistance_coeffs

    def from_pct_strain(self):
        def conv_poly(poly):
            if poly is not None:
                poly = poly * 100**np.arange(len(poly))
            return poly

        def conv_elast(val):
            if val is not None:
                val = 100 * val
            return val

        def conv_therm(val):
            if val is not None:
                val = val / 100
            return val

        # Convert elasticities
        self.outer_elasticity = conv_elast(self.outer_elasticity)
        self.core_elasticity = conv_elast(self.core_elasticity)

        # Convert thermal coeff
        self.outer_thermal_coeff = conv_therm(self.outer_thermal_coeff)
        self.core_thermal_coeff = conv_therm(self.core_thermal_coeff)

        # Convert polynomials
        self.outer_initial_coeffs = conv_poly(self.outer_initial_coeffs)
        self.core_initial_coeffs = conv_poly(self.core_initial_coeffs)
        self.outer_creep_coeffs = conv_poly(self.outer_creep_coeffs)
        self.core_creep_coeffs = conv_poly(self.core_creep_coeffs)

        return self

    def resistance(self, temp):
        return poly.polyval(self.resistance_coeffs, temp)

    def plot_resistances(self, ax=None, symbols={}):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111,
                title=self.name,
                xlabel='Temperature',
                ylabel='Resistance'
            )
            ax.grid()

        sym = dict(
            data='r.',
            curve='b-'
        )
        sym.update(symbols)

        x = self.temperature_data
        y = self.resistance_data

        if x is None or y is None:
            xmin, xmax = 0, 100

        elif sym['data'] is not None:
            ax.plot(x, y, sym['data'])
            xmin, xmax = x.min(), x.max()

        if sym['curve'] is not None:
            x = np.linspace(xmin, xmax, 100)
            y = self.resistance(x)
            ax.plot(x, y, sym['curve'])

        return ax

    def plot_strains(self, ax=None, symbols={}):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111,
                title=self.name,
                xlabel='Strain',
                ylabel='Force'
            )
            ax.grid()

        sym = dict(
            core_initial='b-',
            core_creep='b--',
            outer_initial='r-',
            outer_creep='r--',
            total_initial='g-',
            total_creep='g--',
        )
        sym.update(symbols)

        x = np.linspace(1e-10, 0.006, 100)

        # Initial
        yo, yc, yt = [], [], []

        for s in x:
            odict = self.average_tension('initial', s, self.ref_temperature)
            yo.append(odict['outer']['force'])
            yt.append(odict['force'])

            if odict['core'] is not None:
                yc.append(odict['core']['force'])

        if sym['core_initial'] is not None and len(yc) > 0:
            ax.plot(x, yc, sym['core_initial'], label='Core Initial')

        if sym['outer_initial'] is not None:
            ax.plot(x, yo, sym['outer_initial'], label='Outer Initial')

        if sym['total_initial'] is not None:
            ax.plot(x, yt, sym['total_initial'], label='Composite Initial')

        # Creep
        yo, yc, yt = [], [], []

        for s in x:
            odict = self.average_tension('creep', s, self.ref_temperature)
            yo.append(odict['outer']['force'])
            yt.append(odict['force'])

            if odict['core'] is not None:
                yc.append(odict['core']['force'])

        if sym['core_creep'] is not None and len(yc) > 0:
            ax.plot(x, yc, sym['core_creep'], label='Core Creep')

        if sym['outer_creep'] is not None:
            ax.plot(x, yo, sym['outer_creep'], label='Outer Creep')

        if sym['total_creep'] is not None:
            ax.plot(x, yt, sym['total_creep'], label='Composite Creep')

        ax.legend()

        return ax

    def average_tension(self, curve, strain, to, tc=None, epo=0, epc=0):
        if curve == 'initial':
            core_coeffs = self.core_initial_coeffs
            outer_coeffs = self.outer_initial_coeffs
        elif curve == 'creep':
            core_coeffs = self.core_creep_coeffs
            outer_coeffs = self.outer_creep_coeffs
        else:
            raise ValueError('Curve {!r} invalid.'.format(curve))

        # If core temperature not provided, use outer temperature
        if tc is None:
            tc = to

        outer = mechanics.elastoplastic_force(
            strain=strain,
            temp=to,
            area=self.area,
            coeffs=outer_coeffs,
            tref=self.ref_temperature,
            comp_limit=self.compression_limit,
            linear_strain=self.linear_strain,
            thermal_coeff=self.outer_thermal_coeff,
            elast=self.outer_elasticity,
            plastic_strain=epo
        )

        if self.core_elasticity is None:
            core = None
            force = outer['force']
        else:
            core = mechanics.elastoplastic_force(
                strain=strain,
                temp=tc,
                area=self.area,
                coeffs=core_coeffs,
                tref=self.ref_temperature,
                comp_limit=self.compression_limit,
                linear_strain=self.linear_strain,
                thermal_coeff=self.core_thermal_coeff,
                elast=self.core_elasticity,
                plastic_strain=epc
            )

            force = outer['force'] + core['force']

        return dict(force=force, outer=outer, core=core)
