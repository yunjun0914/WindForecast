import numpy as np

R_SPECIFIC_AIR = 287.05  # J/(kg*K), dry air specific gas constant

ROTOR_DIAMETER_M = {
    "vestas": 126.0,
    "unison": 136.0,
}

# per-turbine rated capacity (info.xlsx 설비용량(MW)), used only to bring the
# flatness collocation loss (which differentiates P_phys, in watts) onto an O(1) scale
SINGLE_TURBINE_CAPACITY_W = {
    "vestas": 3.6e6,
    "unison": 4.2e6,
}


def rotor_area(diameter_m):
    return np.pi * (diameter_m / 2) ** 2


MANUFACTURER_AREA = {name: rotor_area(d) for name, d in ROTOR_DIAMETER_M.items()}


def air_density(temp_k, pressure_pa):
    return pressure_pa / (R_SPECIFIC_AIR * temp_k)
