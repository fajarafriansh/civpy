from ..math import poly

__all__ = [
    'thermal_strain',
    'elastoplastic_force',
]


def thermal_strain(temp, tref, thermal_coeff):
    """
    Returns the thermal strain in a material based on temperature change.

    Parameters
    ----------
    temp : float
        The material temperature.
    tref : float
        The reference temperature.
    thermal_coeff : float
        The thermal expansion coefficient.
    """
    return thermal_coeff * (temp - tref)


def elastoplastic_force(strain, temp, area, elast, coeffs, thermal_coeff, tref,
                        comp_limit=None, linear_strain=None, plastic_strain=0):
    """
    Returns the elastoplastic force

    Parameters
    ----------
    strain : float
        The applied axial strain.
    temp : float
        The material temperature.
    area : float
        The cross sectional area.
    elast : float
        The modulus of elasticity.
    coeffs : float
        The stress-strain polynomial coefficients.
    thermal_coeff : float
        The thermal expansion coefficient.
    tref : float
        The reference temperature.
    comp_limit : float
        The minimum compressive stress in the material. If None, the compressive
        stress is not limited.
    linear_strain : float
        The strain beyond which linear extrapolation will be performed for the
        stress-strain curve. If None, linear extrapolation will not be
        performed.
    plastic_strain : float
        The plastic strain in the material.
    """
    s = strain - thermal_strain(temp, tref, thermal_coeff)
    linear_stress = elast * (s - plastic_strain)

    if linear_strain is None or s <= linear_strain:
        curve_stress = poly.polyval(coeffs, s)
        region = 3
    else:
        m = poly.polyderiv(coeffs, linear_strain)
        b = poly.polyval(coeffs, linear_strain)
        curve_stress = m * (s - linear_strain) + b
        region = 4

    if curve_stress > linear_stress:
        stress = linear_stress
        region = 2
    else:
        stress = curve_stress

    if comp_limit is not None and stress < comp_limit:
        stress = comp_limit
        region = 1

    force = area * stress
    ep = max(plastic_strain, stress / elast)

    return dict(force=force, plastic_strain=ep, region=region)
