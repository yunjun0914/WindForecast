# %%
import pandas as pd
import numpy as np
yy_2024 = pd.read_parquet("./양양_모델/데이터/양양2024.parquet")
yy_2025 = pd.read_parquet("./양양_모델/데이터/양양2025.parquet")
target = pd.read_parquet("./양양_모델/데이터/train_y_yangyang.parquet")

# %%
def target_preprocess(df):
    df = df.copy()
    df["Date/Time"] = df["end_datetime"]
    df["Date/Time"] = df["Date/Time"].dt.tz_localize(None)
    df["Date/Time"] = pd.to_datetime(df["Date/Time"])
    df.index = df["Date/Time"]
    df = df.drop(["end_datetime", "구분", "시간", "energy_mwh", "Date/Time"], axis = 1)
    return df

target = target_preprocess(target)

# %%
def wind(df):
  df = df.copy()
  df["Date/Time"] = df.index
  df["Date/Time"] = pd.to_datetime(df["Date/Time"])
  df.set_index("Date/Time", inplace = True)

  df = df.ffill()
  df = df.fillna(0)

  df["wind_strength_5m"] = (df["usm_5m"] ** 2 + df["vsm_5m"] ** 2) ** 0.5
  df["wind_strength_10m"] = (df["uws_10m"] ** 2 + df["vws_10m"] ** 2) ** 0.5

  df["wind_direction_5m"] = (np.arctan2(df["vsm_5m"], df["usm_5m"]) * 180 / np.pi)  %360
  df["wind_direction_10m"] = (np.arctan2(df["vws_10m"], df["uws_10m"]) * 180 / np.pi) % 360

  df["abs_fvmax"] = abs(df["fvmax_50m"])
  df["abs_fvmin"] = abs(df["fvmin_50m"])

  tdp_c = df["tdp_1p5m"] - 273.15
  vapor_pressure = 6.112 * np.exp((17.67 * tdp_c) / (tdp_c + 243.5))
  df["vapor_pressure"] = vapor_pressure

  return df

# %%
yy_2024 = wind(yy_2024)
yy_2025 = wind(yy_2025)

# %%
yy_all = pd.concat([yy_2024, yy_2025], axis = 0)

# %%
scada = pd.read_parquet("./양양_모델/데이터/scada_yangyang.parquet")

# %%
def scada_preprocess(df ,scada):

    lists = ['WTG03', 'WTG04', 'WTG05', 'WTG06', 'WTG07', 'WTG08', 'WTG09', 'WTG10', 'WTG11', 'WTG12']

    wtg_list = [f"wtg_{i}" for i in range(1, 11)]
    wtgs = {}

    for wtg, c in zip(wtg_list, lists):
        sub = scada[scada["turbine_id"] == c][["wind_speed_mps", "wind_direction_degree", "dt"]]
        wtgs[wtg] = sub
        wtgs[wtg]["dt"] = wtgs[wtg]["dt"].dt.tz_localize(None)
        wtgs[wtg]["Date/Time"] = wtgs[wtg]["dt"]
        wtgs[wtg]["Date/Time"] = pd.to_datetime(wtgs[wtg]["Date/Time"])
        wtgs[wtg].index = wtgs[wtg]["Date/Time"]
        wtgs[wtg]["wind_direction_degree"] = wtgs[wtg]["wind_direction_degree"] % 360
        wtgs[wtg] = wtgs[wtg].drop(["Date/Time", "dt"], axis  =1)
        wtgs[wtg] = wtgs[wtg].ffill()
        wtgs[wtg] = wtgs[wtg].fillna(0)
        wtgs[wtg] = wtgs[wtg].resample("1h").mean()

        wtgs[wtg] = pd.merge(df, wtgs[wtg], left_index= True, right_index= True)
        wtgs[wtg] = wtgs[wtg].ffill()
        wtgs[wtg] = wtgs[wtg].fillna(0)
    

    return wtgs



# %%
yy = scada_preprocess(yy_all, scada)


# %%
trains = {}
tests = {}
wtgs_name = [f"wtg_{i}" for i in range(1, 11)]

for wtg in wtgs_name:
    trains[wtg] = pd.merge(yy[wtg], target, on = "Date/Time")
    yy_0501 = yy[wtg].loc["2024-05-01 01:00:00":"2024-06-01 00:00:00"]
    yy_0701 = yy[wtg].loc["2024-07-01 01:00:00":"2024-08-01 00:00:00"]
    yy_0901 = yy[wtg].loc["2024-09-01 01:00:00":"2024-10-01 00:00:00"]
    yy_1101 = yy[wtg].loc["2024-11-01 01:00:00":"2024-12-01 00:00:00"]
    yy_0101 = yy[wtg].loc["2025-01-01 01:00:00":"2025-02-01 00:00:00"]
    yy_0301 = yy[wtg].loc["2025-03-01 01:00:00":"2025-04-01 00:00:00"]

    all = pd.concat([yy_0501, yy_0701, yy_0901, yy_1101, yy_0101, yy_0301], axis = 0)


    full_idx = pd.date_range("2024-04-01 01:00:00", "2025-04-01 00:00:00", freq = "1h")
    full = pd.DataFrame(index = full_idx)
    full_train = pd.DataFrame(index = full_idx)
    

    tests[wtg] = full.merge(all, how = "left", left_index= True, right_index= True)
    trains[wtg] = full_train.merge(trains[wtg], how = "left", left_index= True, right_index= True)
    tests[wtg] = tests[wtg].fillna(0)
    trains[wtg] = trains[wtg].fillna(0)

# %%
print("all weather & wtg data merged complete!")


