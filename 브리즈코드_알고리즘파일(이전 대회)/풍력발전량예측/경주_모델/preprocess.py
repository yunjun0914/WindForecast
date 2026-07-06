# %%
import numpy as np
def preprocessing(wtgs, target):

    wtg_names = [f"wtg0{i}"for i in range(1, 10)]
    #타겟 & 쓸모없는 변수 & 기계상태 관련 변수 제거(최대한 날씨만으로 예측할수 있게)
    dlt_col = ['elev', 'lssnowr', 'lssnow',
       'Availability\nScheduled Maintenance Time\n[Min.]', 'snowmelt',
       'lsprecipr', 'lsprecip', 'lsmask', 'latitude', 'longitude',
       'Availability\nRequested Shutdown Time\n[Min.]',
       'Availability\nForced Outage Time\n[Min.]', 'Availability\nFull Performance Time\n[Min.]', 'Availability\nTechnical Standby Time\n[Min.]', 'Rotor\nBlade 1 Pos.\n[deg]', 'Rotor\nBlade 2 Pos.\n[deg]',
       'Rotor\nBlade 3 Pos.\n[deg]', 'Rotor\nMotor 1 Pos.\n[deg]',
       'Rotor\nMotor 2 Pos.\n[deg]', 'Rotor\nMotor 3 Pos.\n[deg]',
       'Rotor\nPitch 1 Angle\n[deg]', 'Rotor\nPitch 2 Angle\n[deg]',
       'Rotor\nPitch 3 Angle\n[deg]', 'Hydraulic\nSystem Pressure\n[bar]', "sfr", 
       "energy_kwh", "Energy Production\nActive Energy Production\n[KWh]", 'Grid\nActive Power\n[kW]', 'Grid\nReactive Power\n[kVAr]']
    
    wtgs_train = {}
    wtgs_valid = {}

    for wtg_name in wtg_names:
        #기계 변수 평균처리 & 360도 정규화
        wtgs[wtg_name]["Rotor_Blade_Pos"] = (
            wtgs[wtg_name]['Rotor\nBlade 1 Pos.\n[deg]'] +
            wtgs[wtg_name]['Rotor\nBlade 2 Pos.\n[deg]'] +
            wtgs[wtg_name]['Rotor\nBlade 3 Pos.\n[deg]']) / 3

        wtgs[wtg_name]["Rotor_Motor_Pos"] = (
            wtgs[wtg_name]['Rotor\nMotor 1 Pos.\n[deg]'] +
            wtgs[wtg_name]['Rotor\nMotor 2 Pos.\n[deg]'] +
            wtgs[wtg_name]['Rotor\nMotor 3 Pos.\n[deg]']) / 3

        wtgs[wtg_name]["Rotor_Pitch_Angle"] = (
            wtgs[wtg_name]['Rotor\nPitch 1 Angle\n[deg]'] +
            wtgs[wtg_name]['Rotor\nPitch 2 Angle\n[deg]'] +
            wtgs[wtg_name]['Rotor\nPitch 3 Angle\n[deg]']) / 3
        
        wtgs[wtg_name][['Rotor_Blade_Pos', 'Rotor_Motor_Pos', 'Rotor_Pitch_Angle', 'Yaw\nYaw cable windup\n[deg]']] = ((wtgs[wtg_name][[
            'Rotor_Blade_Pos', 'Rotor_Motor_Pos', 'Rotor_Pitch_Angle', 'Yaw\nYaw cable windup\n[deg]']] % 360) + 360) % 360
        
        # Nacelle Outdoor Temp 이상치 처리
        temp = wtgs[wtg_name]['Nacelle\nOutdoor Temp\n[℃]'] > 50
        wtgs[wtg_name]["Nacelle\nOutdoor Temp\n[℃]"] = np.where(
            temp, np.nan, wtgs[wtg_name]['Nacelle\nOutdoor Temp\n[℃]'])
        
        wtgs[wtg_name]["Nacelle\nOutdoor Temp\n[℃]"] = wtgs[wtg_name]["Nacelle\nOutdoor Temp\n[℃]"].fillna(
        wtgs[wtg_name]["Nacelle\nOutdoor Temp\n[℃]"].median())

        # Nacelle Air Density 이상치 처리
        density = wtgs[wtg_name]['Nacelle\nAir Density\n[kg/㎥]'] < 1
        wtgs[wtg_name]['Nacelle\nAir Density\n[kg/㎥]'] = np.where(density, np.nan, wtgs[wtg_name]['Nacelle\nAir Density\n[kg/㎥]'])


        wtgs[wtg_name]['Nacelle\nAir Density\n[kg/㎥]'] = wtgs[wtg_name]['Nacelle\nAir Density\n[kg/㎥]'].fillna(
        wtgs[wtg_name]['Nacelle\nAir Density\n[kg/㎥]'].median())

        # 풍속 하루단위 평균
        df_rolling_day = wtgs[wtg_name]['Nacelle\nWind Speed\n[m/s]'].rolling(window=24).mean()
        wtgs[wtg_name]['Nacelle\nWind Speed\n[m/s]_mean'] = df_rolling_day
        wtgs[wtg_name]['Nacelle\nWind Speed\n[m/s]_mean'] = wtgs[wtg_name]['Nacelle\nWind Speed\n[m/s]_mean'].fillna(
            wtgs[wtg_name]['Nacelle\nWind Speed\n[m/s]_mean'].median())
        
        wtgs[wtg_name][['abs_usm_5m', 'abs_uws_10m','abs_vsm_5m', 'abs_vws_10m']] = abs(wtgs[wtg_name][['usm_5m', 'uws_10m',
               'vsm_5m', 'vws_10m']])
        
        wtgs[wtg_name] = wtgs[wtg_name].drop(dlt_col, axis =1)

        wtgs_train[wtg_name] = wtgs[wtg_name].loc["2020-01-01":"2022-12-31 23:00:00"]
        wtgs_valid[wtg_name] = wtgs[wtg_name].loc["2023-01-01":"2023-12-31 23:00:00"]
    
    y_train = target.loc["2020-01-01":"2022-12-31 23:00:00"]
    y_valid = target.loc["2023-01-01":"2023-12-31 23:00:00"]

        
    return wtgs_train, wtgs_valid, y_train, y_valid


