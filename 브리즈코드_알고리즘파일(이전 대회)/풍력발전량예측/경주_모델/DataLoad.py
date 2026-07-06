# %%
def merge_with_weahter(df, weather):
    import pandas as pd
    df=  df.copy()
    df["Date/Time"] = pd.to_datetime(df["Date/Time"])
    df.set_index("Date/Time", inplace = True)
    weather = weather.copy()
    weather["Date/Time"] = pd.to_datetime(weather["Date/Time"])
    weather.set_index("Date/Time", inplace = True)
    merged = df.merge(weather, how = "right", on = "Date/Time")
    merged = merged.ffill()
    merged = merged.fillna(0)
    target = merged["energy_kwh"]
    return merged, target

# %%
def DataLoad():
    import pandas as pd
    wtg_list = [f"wtg0{i}"for i in range(1, 10)]
    wtgs = {}
    weather = pd.read_csv("경주_모델/데이터/경주 데이터/경주_2020-2023_날씨.csv")
    test = pd.read_csv("경주_모델/데이터/경주 데이터/경주2024날씨.csv")
    full_idx = pd.date_range("2024-01-01 01:00:00", "2025-01-01 00:00:00", freq = "1h")
    full = pd.DataFrame(index= full_idx)
    test["Date/Time"] = test["datetime"]
    test["Date/Time"] = pd.to_datetime(test["Date/Time"])
    test.set_index("Date/Time", inplace = True)
    test = test.drop("datetime", axis =1)
    test[['abs_usm_5m', 'abs_uws_10m','abs_vsm_5m', 'abs_vws_10m']] = abs(test[['usm_5m', 'uws_10m', 'vsm_5m', 'vws_10m']])
    test = full.merge(test, left_index= True, right_index= True, how = "left")
    test = test.ffill()

    for idx, wtg in enumerate(wtg_list, start = 1):
        df = pd.read_csv(f"경주_모델/데이터/scada 경주/WTG0{idx}_총정보.csv")
        merged, target = merge_with_weahter(df, weather)
        wtgs[wtg] = merged
        print(f"{wtg} merged!")
    print("All DataLoad & Merge Complete!")
    return wtgs, test, target


