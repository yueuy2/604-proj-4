import numpy as np
import os
import glob
import requests
import pandas as pd
from datetime import datetime, timedelta

from matplotlib import pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor

from rich.progress import track

import json
import statsmodels.formula.api as smf


DATA_DIR = 'hrl_load_metered_2016-2025/'
search_path = os.path.join(DATA_DIR, '**', '*.csv')
csv_files = sorted(glob.glob(search_path, recursive=True))
if not csv_files:
    dataframes = {}
else:
    dataframes = {}
    all_dates = []  
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            date_col = df.columns[0]
            parsed_dates = pd.to_datetime(df[date_col], errors='coerce')
            all_dates.append(parsed_dates)
            dataframes[file_name] = df
        except Exception as e:
            pass

if not dataframes:
    pass
else:
    all_dates_combined = pd.concat(all_dates).dropna()
    min_date = all_dates_combined.min()
    max_date = all_dates_combined.max()
    
    start_str = min_date.strftime('%Y-%m-%d')
    end_str = max_date.strftime('%Y-%m-%d')
    
    API_URL = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 39.95,
        "longitude": -75.16,
        "start_date": start_str,
        "end_date": end_str,
        "hourly": "temperature_2m"
    }

    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status() 
        weather_data = response.json()
        hourly_data = weather_data.get('hourly', {})
        if not hourly_data:
            pass
        else:
            weather_df = pd.DataFrame(hourly_data)
            weather_df['time'] = pd.to_datetime(weather_df['time'])
            weather_df = weather_df.set_index('time')
            
            first_df_key = list(dataframes.keys())[0]
            original_df = dataframes[first_df_key]
            date_col = original_df.columns[0]
            original_df[date_col] = pd.to_datetime(original_df[date_col])
            original_df = original_df.set_index(date_col)
            merged_df = original_df.merge(weather_df, left_index=True, right_index=True, how='left')

    except requests.exceptions.RequestException as e:
        pass
    except json.JSONDecodeError:
        pass
    except Exception as e:
        pass

hourly_data = weather_data.get('hourly', {})
if not hourly_data:
    pass
else:
    weather_df = pd.DataFrame(hourly_data)
    weather_df['time'] = pd.to_datetime(weather_df['time'], errors='coerce')
    weather_df = weather_df.dropna(subset=['time']).set_index('time').sort_index()

    merged_dataframes = {}
    for fname, original_df in dataframes.items():
        df_i = original_df.copy()
        date_col = df_i.columns[0]
        df_i[date_col] = pd.to_datetime(df_i[date_col], errors='coerce')
        df_i = df_i.dropna(subset=[date_col]).set_index(date_col).sort_index()

        merged_i = df_i.merge(weather_df, left_index=True, right_index=True, how='left')
        merged_i["source_file"] = fname
        merged_dataframes[fname] = merged_i

    merged_df = (
        pd.concat(merged_dataframes.values(), axis=0, ignore_index=False)
          .rename_axis("timestamp")
          .reset_index()
    )

df = merged_df.copy()

df["ts"] = pd.to_datetime(df["datetime_beginning_ept"], errors="coerce")
df = df.dropna(subset=["ts"]).sort_values(["load_area", "ts"]).reset_index(drop=True)
df["year"] = df["ts"].dt.year
df["hour"] = df["ts"].dt.hour
df["doy"]  = df["ts"].dt.dayofyear  

def add_last_year_feats(frame: pd.DataFrame, n_years: int = 4) -> pd.DataFrame:
    out = frame
    base = (frame[["load_area", "ts", "mw"]]
            .groupby(["load_area", "ts"], as_index=False)["mw"].mean())
    for k in range(1, n_years + 1):
        prev = base.rename(columns={"mw": f"mw_ly{k}"}).copy()
        prev["ts"] = prev["ts"] + pd.DateOffset(years=k)
        out = out.merge(prev, on=["load_area", "ts"], how="left")
    return out

df = add_last_year_feats(df, n_years=4)

def add_lag_by_lookup(frame: pd.DataFrame, hours: int, out_col: str) -> pd.DataFrame:
    key = frame[["load_area", "ts", "mw"]].copy()
    key = key.rename(columns={"mw": out_col})
    key["ts"] = key["ts"] + pd.Timedelta(hours=hours)
    return frame.merge(key, on=["load_area", "ts"], how="left")

for h, c in [(24, "mw_lag24"), (48, "mw_lag48"), (168, "mw_lag168")]:
    df = add_lag_by_lookup(df, hours=h, out_col=c)

HOLD_LO = pd.Timestamp("2025-10-22 00:00:00")
HOLD_HI = pd.Timestamp("2025-11-01 00:00:00")
is_holdout = (df["ts"].ge(HOLD_LO) & df["ts"].lt(HOLD_HI) & df["year"].eq(2025))

valid_df = df.loc[is_holdout].copy()
train_df = df.loc[~is_holdout].copy()

for lag_h, lag_col in [(24, "mw_lag24"), (48, "mw_lag48"), (168, "mw_lag168")]:
    ref_ts = train_df["ts"] - pd.Timedelta(hours=lag_h)
    leak_mask = (ref_ts >= HOLD_LO) & (ref_ts < HOLD_HI)
    train_df.loc[leak_mask, lag_col] = np.nan

feat_cols = ["mw_lag24", "mw_lag48", "mw_lag168"]
train_df = train_df.dropna(subset=feat_cols + ["mw"]).copy()


## Task 1


load_area = [
    "AECO",
    "AEPAPT",
    "AEPIMP",
    "AEPKPT",
    "AEPOPT",
    "AP",
    "BC",
    "CE",
    "DAY",
    "DEOK",
    "DOM",
    "DPLCO",
    "DUQ",
    "EASTON",
    "EKPC",
    "JC",
    "ME",
    "OE",
    "OVEC",
    "PAPWR",
    "PE",
    "PEPCO",
    "PLCO",
    "PN",
    "PS",
    "RECO",
    "SMECO",
    "UGI",
    "VMEU"
]

TRAIN_END = pd.Timestamp("2025-11-17 00:00:00")
ROLL_START_DAY = pd.Timestamp("2025-11-17").normalize()
ROLL_END_DAY   = pd.Timestamp("2025-11-20").normalize()

FEATS = ["mw_lag24", "mw_lag48", "mw_lag168"]

df = merged_df.copy()
df["ts"] = pd.to_datetime(df["datetime_beginning_ept"], errors="coerce")
df = df.dropna(subset=["ts"]).sort_values(["load_area", "ts"]).reset_index(drop=True)
df["year"] = df["ts"].dt.year
df["hour"] = df["ts"].dt.hour
df["doy"]  = df["ts"].dt.dayofyear 
df = df[df["load_area"].isin(load_area)].copy()

def add_lag_by_lookup(frame: pd.DataFrame, hours: int, out_col: str) -> pd.DataFrame:
    key = frame[["load_area", "ts", "mw"]].copy()
    key = key.rename(columns={"mw": out_col})
    key["ts"] = key["ts"] + pd.Timedelta(hours=hours)
    return frame.merge(key, on=["load_area", "ts"], how="left")

for h, c in [(24, "mw_lag24"), (48, "mw_lag48"), (168, "mw_lag168")]:
    df = add_lag_by_lookup(df, hours=h, out_col=c)

train_cut = df.loc[df["ts"] < TRAIN_END].dropna(subset=FEATS + ["mw"]).copy()
X_tr = train_cut[FEATS].to_numpy()
y_tr = train_cut["mw"].to_numpy()

hgb_roll = HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.05,
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=100,
    validation_fraction=0.10,
    max_bins=255,
    min_samples_leaf=50,
    l2_regularization=1.0,
    random_state=42,
    verbose=0
)
hgb_roll.fit(X_tr, y_tr)

roll_hours = pd.date_range(
    ROLL_START_DAY,
    ROLL_END_DAY + pd.Timedelta(days=1),
    freq="H",
    inclusive="left"
)

areas = sorted(df["load_area"].dropna().unique().tolist())

known = {(r.load_area, r.ts): r.mw
         for r in train_cut[["load_area","ts","mw"]].itertuples(index=False)}

def feat_vec(area, ts, known_map):
    l24  = known_map.get((area, ts - pd.Timedelta(hours=24)),  np.nan)
    l48  = known_map.get((area, ts - pd.Timedelta(hours=48)),  np.nan)
    l168 = known_map.get((area, ts - pd.Timedelta(hours=168)), np.nan)
    return np.array([[l24, l48, l168]], dtype=float)

rows = []
for area in areas:
    for ts in roll_hours:
        X = feat_vec(area, ts, known)
        yhat = float(hgb_roll.predict(X)[0])
        rows.append((area, ts, ts.hour, yhat))
        known[(area, ts)] = yhat

pred_consecutive = (pd.DataFrame(rows, columns=["load_area","ts","hour","pred"])
                      .sort_values(["load_area","ts"])
                      .reset_index(drop=True))

eval_slice = df.loc[(df["ts"] >= ROLL_START_DAY) & (df["ts"] < ROLL_END_DAY + pd.Timedelta(days=1)),
                    ["load_area","ts","mw"]]
eval_df = pred_consecutive.merge(eval_slice, on=["load_area","ts"], how="left")

if eval_df["mw"].notna().any():
    def rmse(a,b):
        a,b = np.asarray(a), np.asarray(b)
        return float(np.sqrt(np.mean((a-b)**2)))
    def mae(a,b):
        a,b = np.asarray(a), np.asarray(b)
        return float(np.mean(np.abs(a-b)))
    def mape(a,b):
        a,b = np.asarray(a), np.asarray(b)
        m = a != 0
        return float(np.mean(np.abs((a[m]-b[m])/a[m]))*100.0) if m.any() else np.nan

    rmse(eval_df['mw'].dropna(), eval_df.loc[eval_df['mw'].notna(), 'pred'])
    mae(eval_df['mw'].dropna(), eval_df.loc[eval_df['mw'].notna(), 'pred'])
    mape(eval_df['mw'].dropna(), eval_df.loc[eval_df['mw'].notna(), 'pred'])

pred_2025_11_20 = pred_consecutive[(pred_consecutive["ts"] >= pd.Timestamp("2025-11-20")) &
                                   (pred_consecutive["ts"] <  pd.Timestamp("2025-11-21"))] \
                                  .copy() \
                                  .sort_values(["load_area","ts"])



## Task 2

import numpy as np
import pandas as pd

def robust_sigma(s):
    med = np.median(s)
    mad = np.median(np.abs(s - med))
    if np.isnan(mad) or mad == 0:
        std = np.std(s)
        return float(std if std > 1e-6 else 1.0)
    return float(1.4826 * mad)

train_df = df
X_tr = train_df[["mw_lag24", "mw_lag48", "mw_lag168"]].to_numpy()
y_tr = train_df["mw"].to_numpy()
train_df = train_df.copy()
train_df["pred_tr"] = hgb_roll.predict(X_tr)
train_df["resid"] = train_df["mw"] - train_df["pred_tr"]

sigma_map = (train_df.groupby(["load_area","hour"], observed=True)["resid"]
                        .apply(robust_sigma))
sigma_hour_global = (train_df.groupby("hour", observed=True)["resid"]
                               .apply(robust_sigma))

def get_sigma_vec(area, hours):
    out = []
    for h in hours:
        key = (area, int(h))
        if key in sigma_map.index:
            out.append(float(sigma_map.loc[key]))
        else:
            out.append(float(sigma_hour_global.loc[int(h)]))
    return np.array(out, dtype=float)

ROLL_START_DAY = pd.Timestamp("2025-11-17")
ROLL_END_DAY   = pd.Timestamp("2025-11-20")
FEATS = ["mw_lag24", "mw_lag48", "mw_lag168"]

roll_hours = pd.date_range(ROLL_START_DAY, ROLL_END_DAY + pd.Timedelta(days=1),
                           freq="H", inclusive="left")

areas = sorted(df["load_area"].dropna().unique().tolist())

known = {(r.load_area, r.ts): r.mw
         for r in df.loc[df["ts"] < ROLL_START_DAY, ["load_area","ts","mw"]]
                   .itertuples(index=False)}

def feat_vec(area, ts, known_map):
    l24  = known_map.get((area, ts - pd.Timedelta(hours=24)),  np.nan)
    l48  = known_map.get((area, ts - pd.Timedelta(hours=48)),  np.nan)
    l168 = known_map.get((area, ts - pd.Timedelta(hours=168)), np.nan)
    return np.array([[l24, l48, l168]], dtype=float)

rows = []
for area in areas:
    for ts in roll_hours:
        X1 = feat_vec(area, ts, known)
        yhat = float(hgb_roll.predict(X1)[0])
        rows.append((area, ts, ts.hour, yhat))
        known[(area, ts)] = yhat

pred_consec = (pd.DataFrame(rows, columns=["load_area","ts","hour","pred"])
                 .sort_values(["load_area","ts"])
                 .reset_index(drop=True))

pred_2025_11_20 = pred_consec[(pred_consec["ts"] >= pd.Timestamp("2025-11-20")) &
                              (pred_consec["ts"] <  pd.Timestamp("2025-11-21"))] \
                             .copy()

def soft_argmax(mu, kernel=(1,3,1)):
    k = np.array(kernel, dtype=float)
    sm = np.convolve(mu, k, mode="same")
    return int(np.argmax(sm))

def probabilistic_peak(mu, sigma, n_draws=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    Z = rng.standard_normal((n_draws, len(mu)))
    Y = mu[None, :] + sigma[None, :] * Z
    hmax = np.argmax(Y, axis=1)
    votes = np.zeros(len(mu), dtype=int)
    for h in hmax:
        votes[h] += 1
        if h - 1 >= 0: 
            votes[h - 1] += 1
        if h + 1 < len(mu): 
            votes[h + 1] += 1
    return int(np.argmax(votes))

p20 = pred_2025_11_20.copy()
p20["date"] = p20["ts"].dt.normalize()
p20["hour"] = p20["hour"].astype(int)

rows = []
for area, g in p20.groupby("load_area", sort=True):
    gg = g.sort_values("hour")
    mu  = gg["pred"].to_numpy()
    sig = get_sigma_vec(area, gg["hour"].to_numpy())

    idx_arg  = int(np.argmax(mu))
    idx_soft = soft_argmax(mu, kernel=(1,3,1))
    idx_prob = probabilistic_peak(mu, sig, n_draws=2000)

    peak_arg = int(gg["hour"].iloc[idx_arg])
    peak_soft= int(gg["hour"].iloc[idx_soft])
    peak_prob= int(gg["hour"].iloc[idx_prob])

    rows.append({
        "load_area": area,
        "date":      pd.Timestamp("2025-11-20").date(),
        "peak_hour_argmax": peak_arg,
        "peak_hour_soft":   peak_soft,
        "peak_hour_prob":   peak_prob
    })

peak_hours_2025_11_20 = (pd.DataFrame(rows)
                         .sort_values("load_area")
                         .reset_index(drop=True))
peak_hours_2025_11_20["peak_hour_submit"] = peak_hours_2025_11_20["peak_hour_prob"]
peak_hours_pred = peak_hours_2025_11_20["peak_hour_prob"]


## Task 3


import numpy as np
import pandas as pd

df = merged_df.copy()
if 'load_area' in globals():
    df = df[df["load_area"].isin(load_area)].copy()

df["ts"]   = pd.to_datetime(df["datetime_beginning_ept"], errors="coerce")
df         = df.dropna(subset=["ts"]).sort_values(["load_area","ts"]).reset_index(drop=True)
df["date"] = df["ts"].dt.normalize()

daily = (df.groupby(["load_area","date"], observed=True)["mw"]
           .max().rename("mw_peak").reset_index())

K_PAST   = 3
WIN_START = pd.Timestamp("2025-11-20").normalize()
WIN_END   = pd.Timestamp("2025-11-30").normalize()

def week_start_monday(dt):
    return pd.Timestamp(dt).to_period("W-MON").start_time.normalize()

target_dates_2025 = pd.date_range(WIN_START, WIN_END - pd.Timedelta(days=1), freq="D")

week_starts_2025 = sorted({week_start_monday(d) for d in target_dates_2025})

pred_rows = []

for W25_START in week_starts_2025:
    W25_END = W25_START + pd.Timedelta(days=7)
    W24_START = W25_START - pd.DateOffset(years=1)
    W24_END   = W24_START + pd.Timedelta(days=7)

    d24 = daily[(daily["date"] >= W24_START) & (daily["date"] < W24_END)].copy()
    if d24.empty:
        continue

    areas = sorted(d24["load_area"].unique())

    week25_days = [W25_START + pd.Timedelta(days=i) for i in range(7)]
    grid = pd.MultiIndex.from_product([areas, week25_days], names=["load_area","date"]).to_frame(index=False)

    map_pos_24 = pd.DataFrame({
        "date": [W24_START + pd.Timedelta(days=i) for i in range(7)],
        "pos":  np.arange(7)
    })
    map_pos_25 = pd.DataFrame({
        "date": [W25_START + pd.Timedelta(days=i) for i in range(7)],
        "pos":  np.arange(7)
    })

    d24  = d24.merge(map_pos_24, on="date", how="left")
    grid = grid.merge(map_pos_25, on="date", how="left")

    topK_2024 = (d24.sort_values(["load_area","mw_peak"], ascending=[True, False])
                   .groupby("load_area", as_index=False).head(K_PAST)
                   [["load_area","pos"]]
                   .groupby("load_area")["pos"]
                   .apply(lambda s: sorted(pd.unique(s)))
                   .rename("pos_set").reset_index())

    grid = grid.merge(topK_2024, on="load_area", how="left")
    grid["pred_peakday"] = grid.apply(
        lambda r: int(isinstance(r["pos_set"], list) and (r["pos"] in r["pos_set"])),
        axis=1
    )

    grid = grid.merge(daily[["load_area","date","mw_peak"]], on=["load_area","date"], how="left")

    grid = grid[(grid["date"] >= WIN_START) & (grid["date"] < WIN_END)].copy()

    pred_rows.append(grid[["load_area","date","mw_peak","pred_peakday"]])

if pred_rows:
    pred_peakdays_2025 = (pd.concat(pred_rows, ignore_index=True)
                            .sort_values(["load_area","date"])
                            .reset_index(drop=True))
else:
    pred_peakdays_2025 = pd.DataFrame(columns=["load_area","date","mw_peak","pred_peakday"])

peakday_pred = pred_peakdays_2025[pred_peakdays_2025["date"] == "2025-11-20"]["pred_peakday"]

first_two = np.append(pred_2025_11_20["pred"],peak_hours_pred)

df_out = pd.DataFrame(
    {"2025-11-20": np.append(first_two,peakday_pred)}
)

df_out.to_csv("pred_2025_11_20.csv", index=False)
print(list(df_out.iloc[0:]))