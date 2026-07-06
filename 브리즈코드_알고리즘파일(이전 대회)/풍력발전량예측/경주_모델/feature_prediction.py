# %%
import pandas as pd
import numpy as np

wtg_cols = ['Generator\nAverage Winding Temp.\n[℃]', 'Nacelle\nAir Density\n[kg/㎥]',
            'Nacelle\nNacelle Position\n[deg]', 'Nacelle\nOutdoor Temp\n[℃]',
            'Nacelle\nWind Direction\n[deg]', 'Nacelle\nWind Speed\n[m/s]', 'Nacelle\nWind Speed\n[m/s]_mean']

#피쳐    
weather_cols = ['dswrf', 'fvmax_50m', 'fvmin_50m',
               'lhnf', 'maxsa_1p5m', 'mcc', 'mgws_0m', 'p', 'pblh', 'pmsl', 'rh_1p5m',
               'sh_1p5m', 'ta', 'ta_1p5m', 'tdp_1p5m', 'usm_5m', 'uws_10m',
               'vsm_5m', 'vws_10m', 'wind_direction_10m', 'wind_direction_5m', 'wind_strength_10m',
               'wind_strength_5m', "vapor_pressure", "abs_fvmax", "abs_fvmin", 'abs_usm_5m', 'abs_uws_10m','abs_vsm_5m', 'abs_vws_10m']
    

def train_target_split(wtgs, test):
    wtg_names = [f"wtg0{i}" for i in range(1, 10)]
    wtgs_train = {}
    wtgs_target = {}
    tests = {}

    for c in wtg_names:
        # 분리
        wtgs_target[c] = wtgs[c][wtg_cols]
        wtgs_train[c] = wtgs[c][weather_cols]
        tests[c] = test[weather_cols]
        
    return wtgs_train, wtgs_target, tests



