import argparse
import sys
import os
import glob
import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt  # not actually used in core pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib


# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------
DATA_DIR = "hrl_load_metered_2016-2025/"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

WEATHER_LAT = 39.95
WEATHER_LON = -75.16

LOAD_AREAS = [
    "AECO", "AEPAPT", "AEPIMP", "AEPKPT", "AEPOPT", "AP", "BC", "CE", "DAY",
    "DEOK", "DOM", "DPLCO", "DUQ", "EASTON", "EKPC", "JC", "ME", "OE", "OVEC",
    "PAPWR", "PE", "PEPCO", "PLCO", "PN", "PS", "RECO", "SMECO", "UGI", "VMEU",
]

FEATS = ["mw_lag24", "mw_lag48", "mw_lag168"]

# training window for Task 1 / Task 2 model
TRAIN_END = pd.Timestamp("2025-11-17 00:00:00")

# rolling prediction window (for Nov 20 predictions, we need Nov 17–20 history)
ROLL_START_DAY = pd.Timestamp("2025-11-17").normalize()
ROLL_END_DAY   = pd.Timestamp("2025-11-20").normalize()

TARGET_DATE = pd.Timestamp("2025-11-20").normalize()

# filenames for saved models / objects
HGB_MODEL_PATH = MODEL_DIR / "hgb_roll.pkl"
SIGMA_MAP_PATH = MODEL_DIR / "sigma_map.pkl"
SIGMA_HOUR_GLOBAL_PATH = MODEL_DIR / "sigma_hour_global.pkl"


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------
def load_pjm_and_weather() -> pd.DataFrame:
    """
    Load all PJM CSVs under DATA_DIR, infer min/max dates, download
    hourly temperature from Open-Meteo, and return a merged dataframe.
    """
    search_path = os.path.join(DATA_DIR, "**", "*.csv")
    csv_files = sorted(glob.glob(search_path, recursive=True))

    if not csv_files:
        raise RuntimeError(f"No CSV files found in {DATA_DIR}")

    dataframes = {}
    all_dates = []

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            date_col = df.columns[0]
            parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
            all_dates.append(parsed_dates)
            dataframes[file_name] = df
        except Exception:
            # skip unreadable files
            continue

    if not dataframes:
        raise RuntimeError("No valid CSV files loaded from DATA_DIR.")

    all_dates_combined = pd.concat(all_dates).dropna()
    min_date = all_dates_combined.min()
    max_date = all_dates_combined.max()

    start_str = min_date.strftime("%Y-%m-%d")
    end_str = max_date.strftime("%Y-%m-%d")

    # Download hourly temperature
    API_URL = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": WEATHER_LAT,
        "longitude": WEATHER_LON,
        "start_date": start_str,
        "end_date": end_str,
        "hourly": "temperature_2m",
    }

    response = requests.get(API_URL, params=params)
    response.raise_for_status()
    weather_data = response.json()

    hourly_data = weather_data.get("hourly", {})
    if not hourly_data:
        raise RuntimeError("Weather API returned no hourly data.")

    weather_df = pd.DataFrame(hourly_data)
    weather_df["time"] = pd.to_datetime(weather_df["time"], errors="coerce")
    weather_df = weather_df.dropna(subset=["time"]).set_index("time").sort_index()

    # Merge each PJM file with weather
    merged_dataframes = {}
    for fname, original_df in dataframes.items():
        df_i = original_df.copy()
        date_col = df_i.columns[0]
        df_i[date_col] = pd.to_datetime(df_i[date_col], errors="coerce")
        df_i = df_i.dropna(subset=[date_col]).set_index(date_col).sort_index()

        merged_i = df_i.merge(weather_df, left_index=True, right_index=True, how="left")
        merged_i["source_file"] = fname
        merged_dataframes[fname] = merged_i

    merged_df = (
        pd.concat(merged_dataframes.values(), axis=0, ignore_index=False)
        .rename_axis("timestamp")
        .reset_index()
    )

    return merged_df


def add_lag_by_lookup(frame: pd.DataFrame, hours: int, out_col: str) -> pd.DataFrame:
    key = frame[["load_area", "ts", "mw"]].copy()
    key = key.rename(columns={"mw": out_col})
    key["ts"] = key["ts"] + pd.Timedelta(hours=hours)
    return frame.merge(key, on=["load_area", "ts"], how="left")


def robust_sigma(s: pd.Series) -> float:
    med = np.median(s)
    mad = np.median(np.abs(s - med))
    if np.isnan(mad) or mad == 0:
        std = np.std(s)
        return float(std if std > 1e-6 else 1.0)
    return float(1.4826 * mad)


def get_sigma_vec(area, hours, sigma_map, sigma_hour_global):
    out = []
    for h in hours:
        key = (area, int(h))
        if key in sigma_map.index:
            out.append(float(sigma_map.loc[key]))
        else:
            out.append(float(sigma_hour_global.loc[int(h)]))
    return np.array(out, dtype=float)


def soft_argmax(mu, kernel=(1, 3, 1)):
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


def week_start_monday(dt):
    return pd.Timestamp(dt).to_period("W-MON").start_time.normalize()


# ---------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------
def run_clean():
    """
    Delete all generated artifacts except raw data and code.
    """
    if MODEL_DIR.exists():
        for p in MODEL_DIR.glob("*"):
            p.unlink()
    print("Cleaned model artifacts.", file=sys.stderr)


def run_train():
    """
    Train models / statistics and save them to disk.
    """
    print("Loading PJM + weather data...", file=sys.stderr)
    merged_df = load_pjm_and_weather()

    df = merged_df.copy()
    df = df[df["load_area"].isin(LOAD_AREAS)].copy()

    df["ts"] = pd.to_datetime(df["datetime_beginning_ept"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values(["load_area", "ts"]).reset_index(drop=True)
    df["year"] = df["ts"].dt.year
    df["hour"] = df["ts"].dt.hour
    df["doy"] = df["ts"].dt.dayofyear

    # Add lags
    for h, c in [(24, "mw_lag24"), (48, "mw_lag48"), (168, "mw_lag168")]:
        df = add_lag_by_lookup(df, hours=h, out_col=c)

    # -----------------------------------------------------------------
    # Train HGB model (Task 1 / Task 2)
    # -----------------------------------------------------------------
    print("Training HistGradientBoostingRegressor...", file=sys.stderr)
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
        verbose=0,
    )
    hgb_roll.fit(X_tr, y_tr)

    # -----------------------------------------------------------------
    # Train residual sigma maps (Task 2)
    # -----------------------------------------------------------------
    print("Computing residual sigmas...", file=sys.stderr)
    X_tr_all = df[FEATS].to_numpy()
    y_tr_all = df["mw"].to_numpy()
    df_resid = df.copy()
    df_resid["pred_tr"] = hgb_roll.predict(X_tr_all)
    df_resid["resid"] = df_resid["mw"] - df_resid["pred_tr"]

    sigma_map = (
        df_resid.groupby(["load_area", "hour"], observed=True)["resid"]
        .apply(robust_sigma)
    )
    sigma_hour_global = (
        df_resid.groupby("hour", observed=True)["resid"]
        .apply(robust_sigma)
    )

    # Save all artifacts
    joblib.dump(hgb_roll, HGB_MODEL_PATH)
    joblib.dump(sigma_map, SIGMA_MAP_PATH)
    joblib.dump(sigma_hour_global, SIGMA_HOUR_GLOBAL_PATH)

    print(f"Saved model to {HGB_MODEL_PATH}", file=sys.stderr)
    print("Training complete.", file=sys.stderr)


def run_predictions():
    """
    Load models from disk, make predictions for 2025-11-20, and print
    a single CSV row:
    "YYYY-MM-DD", <loads>, <peak hours>, <peak-day flags>
    """
    # -----------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------
    if not HGB_MODEL_PATH.exists():
        raise RuntimeError("Model file not found; run `make train` first.")

    hgb_roll = joblib.load(HGB_MODEL_PATH)
    sigma_map = joblib.load(SIGMA_MAP_PATH)
    sigma_hour_global = joblib.load(SIGMA_HOUR_GLOBAL_PATH)

    # -----------------------------------------------------------------
    # Recompute features from raw data (same as in training)
    # -----------------------------------------------------------------
    merged_df = load_pjm_and_weather()
    df = merged_df.copy()
    df = df[df["load_area"].isin(LOAD_AREAS)].copy()

    df["ts"] = pd.to_datetime(df["datetime_beginning_ept"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values(["load_area", "ts"]).reset_index(drop=True)
    df["year"] = df["ts"].dt.year
    df["hour"] = df["ts"].dt.hour
    df["doy"] = df["ts"].dt.dayofyear

    for h, c in [(24, "mw_lag24"), (48, "mw_lag48"), (168, "mw_lag168")]:
        df = add_lag_by_lookup(df, hours=h, out_col=c)

    # -----------------------------------------------------------------
    # Task 1: rolling hourly predictions up to 2025-11-20
    # -----------------------------------------------------------------
    roll_hours = pd.date_range(
        ROLL_START_DAY,
        ROLL_END_DAY + pd.Timedelta(days=1),
        freq="H",
        inclusive="left",
    )
    areas = sorted(df["load_area"].dropna().unique().tolist())

    # known loads before the rolling window
    known = {
        (r.load_area, r.ts): r.mw
        for r in df.loc[df["ts"] < ROLL_START_DAY, ["load_area", "ts", "mw"]]
        .itertuples(index=False)
    }

    def feat_vec(area, ts, known_map):
        l24 = known_map.get((area, ts - pd.Timedelta(hours=24)), np.nan)
        l48 = known_map.get((area, ts - pd.Timedelta(hours=48)), np.nan)
        l168 = known_map.get((area, ts - pd.Timedelta(hours=168)), np.nan)
        return np.array([[l24, l48, l168]], dtype=float)

    rows = []
    for area in areas:
        for ts in roll_hours:
            X1 = feat_vec(area, ts, known)
            yhat = float(hgb_roll.predict(X1)[0])
            rows.append((area, ts, ts.hour, yhat))
            # feed prediction forward
            known[(area, ts)] = yhat

    pred_consec = (
        pd.DataFrame(rows, columns=["load_area", "ts", "hour", "pred"])
        .sort_values(["load_area", "ts"])
        .reset_index(drop=True)
    )

    # slice Nov 20
    pred_2025_11_20 = (
        pred_consec[
            (pred_consec["ts"] >= TARGET_DATE)
            & (pred_consec["ts"] < TARGET_DATE + pd.Timedelta(days=1))
        ]
        .copy()
        .sort_values(["load_area", "ts"])
    )

    # hourly predictions, rounded to nearest integer
    hourly_pred = np.rint(pred_2025_11_20["pred"].to_numpy()).astype(int)

    # -----------------------------------------------------------------
    # Task 2: peak-hour prediction for Nov 20
    # -----------------------------------------------------------------
    p20 = pred_2025_11_20.copy()
    p20["date"] = p20["ts"].dt.normalize()
    p20["hour"] = p20["hour"].astype(int)

    peak_rows = []
    for area, g in p20.groupby("load_area", sort=True):
        gg = g.sort_values("hour")
        mu = gg["pred"].to_numpy()
        sig = get_sigma_vec(area, gg["hour"].to_numpy(), sigma_map, sigma_hour_global)

        idx_prob = probabilistic_peak(mu, sig, n_draws=2000)
        peak_prob = int(gg["hour"].iloc[idx_prob])

        peak_rows.append({"load_area": area, "peak_hour_prob": peak_prob})

    peak_hours_df = (
        pd.DataFrame(peak_rows).sort_values("load_area").reset_index(drop=True)
    )
    peak_hours_pred = peak_hours_df["peak_hour_prob"].to_numpy(dtype=int)

    # -----------------------------------------------------------------
    # Task 3: peak-day indicator for Nov 20 (rule-based, no model)
    # -----------------------------------------------------------------
    df_daily = df.copy()
    df_daily["date"] = df_daily["ts"].dt.normalize()
    daily = (
        df_daily.groupby(["load_area", "date"], observed=True)["mw"]
        .max()
        .rename("mw_peak")
        .reset_index()
    )

    K_PAST = 3
    WIN_START = pd.Timestamp("2025-11-20").normalize()
    WIN_END = pd.Timestamp("2025-11-30").normalize()

    target_dates_2025 = pd.date_range(
        WIN_START, WIN_END - pd.Timedelta(days=1), freq="D"
    )
    week_starts_2025 = sorted({week_start_monday(d) for d in target_dates_2025})

    pred_rows = []
    for W25_START in week_starts_2025:
        W25_END = W25_START + pd.Timedelta(days=7)
        W24_START = W25_START - pd.DateOffset(years=1)
        W24_END = W24_START + pd.Timedelta(days=7)

        d24 = daily[(daily["date"] >= W24_START) & (daily["date"] < W24_END)].copy()
        if d24.empty:
            continue

        areas_week = sorted(d24["load_area"].unique())
        week25_days = [W25_START + pd.Timedelta(days=i) for i in range(7)]
        grid = pd.MultiIndex.from_product(
            [areas_week, week25_days], names=["load_area", "date"]
        ).to_frame(index=False)

        map_pos_24 = pd.DataFrame(
            {
                "date": [W24_START + pd.Timedelta(days=i) for i in range(7)],
                "pos": np.arange(7),
            }
        )
        map_pos_25 = pd.DataFrame(
            {
                "date": [W25_START + pd.Timedelta(days=i) for i in range(7)],
                "pos": np.arange(7),
            }
        )

        d24 = d24.merge(map_pos_24, on="date", how="left")
        grid = grid.merge(map_pos_25, on="date", how="left")

        topK_2024 = (
            d24.sort_values(["load_area", "mw_peak"], ascending=[True, False])
            .groupby("load_area", as_index=False)
            .head(K_PAST)[["load_area", "pos"]]
            .groupby("load_area")["pos"]
            .apply(lambda s: sorted(pd.unique(s)))
            .rename("pos_set")
            .reset_index()
        )

        grid = grid.merge(topK_2024, on="load_area", how="left")
        grid["pred_peakday"] = grid.apply(
            lambda r: int(isinstance(r["pos_set"], list) and (r["pos"] in r["pos_set"])),
            axis=1,
        )

        grid = grid.merge(
            daily[["load_area", "date", "mw_peak"]],
            on=["load_area", "date"],
            how="left",
        )
        grid = grid[(grid["date"] >= WIN_START) & (grid["date"] < WIN_END)].copy()

        pred_rows.append(grid[["load_area", "date", "mw_peak", "pred_peakday"]])

    if pred_rows:
        pred_peakdays_2025 = (
            pd.concat(pred_rows, ignore_index=True)
            .sort_values(["load_area", "date"])
            .reset_index(drop=True)
        )
    else:
        pred_peakdays_2025 = pd.DataFrame(
            columns=["load_area", "date", "mw_peak", "pred_peakday"]
        )

    peakday_pred = (
        pred_peakdays_2025[pred_peakdays_2025["date"] == TARGET_DATE][
            ["load_area", "pred_peakday"]
        ]
        .sort_values("load_area")["pred_peakday"]
        .to_numpy(dtype=int)
    )

    # -----------------------------------------------------------------
    # Assemble final output row and print
    # -----------------------------------------------------------------
    # "2025-11-20", L1_00, ..., L29_23, PH_1, ..., PH_29, PD_1, ..., PD_29
    row_values = np.concatenate(
        [hourly_pred.astype(int), peak_hours_pred.astype(int), peakday_pred.astype(int)]
    )

    # Print as a single CSV line to stdout, with date in quotes
    parts = ['"2025-11-20"'] + [str(int(v)) for v in row_values]
    print(",".join(parts))


def run_rawdata():
    """
    Delete and re-download *only* the weather data–dependent merged file,
    or simply rely on run_train / run_predictions to re-fetch.

    For now we just print a message; raw PJM CSVs are assumed to already
    be present under DATA_DIR.
    """
    print(
        "run_rawdata: PJM CSVs are expected in "
        f"{DATA_DIR}. Weather data is fetched on the fly.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline entrypoint: clean, predictions, train, or rawdata"
    )
    parser.add_argument(
        "task",
        choices=["clean", "predictions", "train", "rawdata"],
        help="Which step of the pipeline to run",
    )
    args = parser.parse_args()

    if args.task == "clean":
        run_clean()
    elif args.task == "predictions":
        run_predictions()
    elif args.task == "train":
        run_train()
    elif args.task == "rawdata":
        run_rawdata()
    else:
        print("Unrecognized command.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
