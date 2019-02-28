"""
Copyright (c) 2019, Matt Pewsey
"""

import propy

__all__ = ['WeatherCase']


class WeatherCase(object):
    """
    A class representing weather conditions.

    Parameters
    ----------
    name : str
        The name of the weather case.
    wind_velocity : float
        The wind velocity.
    wind_press_coeff : float
        The wind pressure coefficient for converting wind velocity to pressure.
    ice_thickness : float
        The ice thickness.
    ice_density : float
        The ice ice density.
    """
    # Custom properties
    name = propy.str_property('name')

    def __init__(self, name,
                 wind_velocity=0, wind_press_coeff=0.6125,
                 ice_thickness=0, ice_density=8954):
        self.name = name

        # Wind
        self.wind_velocity = wind_velocity
        self.wind_press_coeff = wind_press_coeff

        # Ice
        self.ice_thickness = ice_thickness
        self.ice_density = ice_density

    def wind_pressure(self):
        """Returns the wind pressure."""
        return self.wind_press_coeff * self.wind_velocity**2
