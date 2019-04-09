from ..structures import LoadCase, WeatherCase


def nesc_heavy(temps=None):
    wc = [
        WeatherCase('0', wire_temperature=0, ice_thickness=0.5, wind_velocity=4),
        WeatherCase('0', wire_temperature=0, ice_thickness=0.5, wind_velocity=4),
        WeatherCase('0', wire_temperature=0, ice_thickness=0.5, wind_velocity=4),
        WeatherCase('0', wire_temperature=0, ice_thickness=0.5, wind_velocity=4),
    ]

    lc = [
        LoadCase('0', wire_condition='initial', weather=wc[0]),
        LoadCase('0', wire_condition='initial', weather=wc[1]),
        LoadCase('0', wire_condition='creep', weather=wc[2]),
        LoadCase('0', wire_condition='creep', weather=wc[3]),
    ]

    constraints = [
        {'tmax': 0.60, 'load_case': lc[0]},
        {'tmax': 0.35, 'load_case': lc[1]},
        {'tmax': 0.25, 'load_case': lc[2]},
    ]

    loads = [lc[3]]
    stringing = []

    if temps is None:
        temps = range(0, 101, 5)

    for temp in temps:
        weather = WeatherCase('', wire_temperature=temp)
        load_case = LoadCase('', weather=weather)
        stringing.append(load_case)

    return dict(constraints=constraints, loads=loads, stringing=stringing)
