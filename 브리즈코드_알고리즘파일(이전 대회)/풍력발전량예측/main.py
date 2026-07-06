# %%
import 경주_main_rf

# %%
import 양양_영덕_예측

# %%
import pandas as pd

yy = pd.read_csv("양양_모델/양양_예측/양양_baseline.csv")
yd = pd.read_csv("영덕_모델/영덕_예측/영덕_baseline.csv")
gj_df = pd.read_csv("경주_모델/경주_예측/경주.csv")

# %%

gj_df["plant_name"] = "경주풍력"
gj_df["Unnamed: 0"] = pd.to_datetime(gj_df["Unnamed: 0"])
gj_df["start_datetime"] = gj_df["Unnamed: 0"] - pd.Timedelta(hours = 1)
gj_df["end_datetime"] = gj_df["Unnamed: 0"]
gj_df["yield_kwh"] = gj_df["pred"]

gj = gj_df.drop(["Unnamed: 0", "pred"], axis = 1)


# %%
import matplotlib.pyplot as plt
plt.figure(figsize = (12,4))
plt.plot(gj["end_datetime"], gj["yield_kwh"], label = "pred")
plt.legend()
plt.show()

# %%
pred = pd.concat([gj, yy, yd], axis= 0)

pred

# %%
pred = pred.reset_index(drop = True)

# %%
pred.to_csv("result.csv", index = False)


