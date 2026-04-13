import pandas as pd
import numpy as np
import re
import os

# -------------------------
# Load + clean L2 ticks
# -------------------------
def _read_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _quick_missing_check(df):
    has_missing = df.isnull().any().any()
    if has_missing:
        missing_cols = df.columns[df.isnull().any()].tolist()
        print(f"Missing values found in {len(missing_cols)} columns: {missing_cols}")
        return True
    else:
        print("No missing values found.")
        return False


def infer_missing_sides_with_lee_ready(df):
    original_order = df.index
    df['side'] = df['side'].astype(str).str.upper().str.strip()
    need_infer_mask = df['side'].isin(['', '-', 'N/A', 'NAN', 'NONE', 'UNKNOWN'])

    if not need_infer_mask.any():
        return df

    df = df.sort_values("timestamp").copy()
    df['mid_price'] = (df['bid1'] + df['ask1']) / 2
    df['price_change'] = df['last'].diff()
    inferred = pd.Series('', index=df.index)

    mask_up = df['price_change'] > 0
    mask_down = df['price_change'] < 0
    mask_stable_above = (df['price_change'] == 0) & (df['last'] > df['mid_price'])
    mask_stable_below = (df['price_change'] == 0) & (df['last'] < df['mid_price'])
    mask_stable_equal = (df['price_change'] == 0) & (df['last'] == df['mid_price'])

    inferred[mask_up] = 'B'
    inferred[mask_down] = 'S'
    inferred[mask_stable_above] = 'B'
    inferred[mask_stable_below] = 'S'

    if mask_stable_equal.any():
        valid_sides = df['side'].where(~df['side'].isin(['', '-', 'N/A', 'NAN', 'NONE', 'UNKNOWN']))
        filled_sides = valid_sides.ffill()
        inferred[mask_stable_equal] = filled_sides[mask_stable_equal]

    inferred = inferred.fillna('U')
    df.loc[need_infer_mask, 'side'] = inferred[need_infer_mask]
    df = df.drop(columns=['mid_price', 'price_change'])
    df = df.loc[original_order]
    return df


def clean_price_anomalies(df, max_pct_change=0.2):
    df_cleaned = df.copy()
    pct_change = df_cleaned['last'].pct_change().abs()
    anomaly_mask = (pct_change > max_pct_change) | (df_cleaned['last'] == 0)

    print(f"Found {anomaly_mask.sum()} anomalies (> {max_pct_change * 100}% change)")
    df_cleaned.loc[anomaly_mask, 'last'] = np.nan
    df_cleaned['last'] = df_cleaned['last'].ffill().bfill()
    df['last'] = df_cleaned['last']
    return df


def load_l2_ticks(file_path: str) -> pd.DataFrame:
    raw = _read_csv(file_path)
    time_col = raw.columns[0]
    raw = raw[raw[time_col].astype(str) != time_col].copy()
    cols = list(raw.columns)

    rename_map = {
        cols[0]: "time",
        cols[1]: "last",
        cols[2]: "side",
        cols[3]: "trade_vol",
        cols[4]: "trade_cnt",
        cols[5]: "trade_amt",
    }
    df = raw.rename(columns=rename_map)

    for c in ["last", "trade_vol", "trade_cnt", "trade_amt"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    date_match = re.search(r'\\(\d{8})\\', file_path)
    if date_match:
        date_str = date_match.group(1)
        file_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        print(file_date)
        df["timestamp"] = pd.to_datetime(file_date + " " + df["time"].astype(str),
                                         format='%Y-%m-%d %H:%M:%S',
                                         errors="coerce")
    else:
        print(f"Warning: Could not extract date from path: {file_path}")

    ob_cols = cols[6:]
    for lvl in range(1, 6):
        bid_px_col = ob_cols[(lvl - 1) * 2 + 0]
        bid_sz_col = ob_cols[(lvl - 1) * 2 + 1]
        ask_px_col = ob_cols[10 + (lvl - 1) * 2 + 0]
        ask_sz_col = ob_cols[10 + (lvl - 1) * 2 + 1]

        df[f"bid{lvl}"] = pd.to_numeric(raw[bid_px_col], errors="coerce")
        df[f"bid{lvl}_sz"] = pd.to_numeric(raw[bid_sz_col], errors="coerce")
        df[f"ask{lvl}"] = pd.to_numeric(raw[ask_px_col], errors="coerce")
        df[f"ask{lvl}_sz"] = pd.to_numeric(raw[ask_sz_col], errors="coerce")

    df = df.dropna(subset=["timestamp", "last", "bid1", "ask1"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = infer_missing_sides_with_lee_ready(df)

    sz_cols = [col for col in df.columns if '_sz' in col]
    for col in sz_cols:
        df[col] = df[col].replace(0, pd.NA)
        df[col] = df[col].ffill()
        df[col] = df[col].fillna(0)
    df = df.groupby('timestamp').last().reset_index()

    keep_cols = ["timestamp", "time", "last", "side", "trade_vol", "trade_cnt", "trade_amt"]
    for lvl in range(1, 6):
        keep_cols.extend([f"bid{lvl}", f"bid{lvl}_sz", f"ask{lvl}", f"ask{lvl}_sz"])

    df = df[keep_cols]
    print("After data cleaning...")
    _quick_missing_check(df)
    df = df.iloc[60:-60].reset_index(drop=True)
    return df


def load_all_days_simple(csv_file, days_list=None):
    if days_list is None:
        days_list = ['20250401', '20250402', '20250403', '20250407', '20250408', '20250409', '20250410']

    all_days_data = []
    for day in days_list:
        if day == '20250410':
            data_dir = r"F:\HKdata\hk 10\_data\stock\20250410\hk"
        else:
            day_num = int(day[-2:])
            folder_name = f"hk 0{day_num}" if 1 <= day_num <= 9 else f"hk {day_num}"
            data_dir = os.path.join(r"F:\HKdata", folder_name, "_data", "stock", day, "hk")

        file_path = os.path.join(data_dir, csv_file)
        if not os.path.exists(file_path):
            print(f"⚠️ File does not exist: {file_path}")
            continue

        df = load_l2_ticks(file_path)
        if df is not None and len(df) > 0:
            all_days_data.append(df)
        else:
            print(f"⚠️ File {file_path} is empty or failed to load")

    if not all_days_data:
        print(f"❌ Stock {csv_file} has no available data")
        return None

    combined_df = pd.concat(all_days_data, axis=0)
    combined_df.sort_values('timestamp', inplace=True)
    combined_df = combined_df.reset_index(drop=True)
    print(f"✅ Successfully loaded {csv_file}: {len(combined_df)} rows")
    return combined_df


def load_all(stock):
    my_days = ['20250401', '20250402', '20250403', '20250407', '20250408', '20250409', '20250410']
    stock_code = stock[:-4]

    print(f"📂 Loading stock {stock_code}...")
    df = load_all_days_simple(stock, my_days)

    if df is None:
        print(f"❌ Stock {stock_code} loading failed, no data available")
        return None

    if len(df) < 10:
        print(f"❌ Stock {stock_code} insufficient data ({len(df)} rows)")
        return None

    print(f"✅ Stock {stock_code} loading complete: {len(df)} rows")
    return df


def fill_missing_features(df):
    df = df.ffill()
    df = df.bfill()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def add_l2_and_orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mid"] = (df["bid1"] + df["ask1"]) / 2.0
    df["spread"] = df["ask1"] - df["bid1"]
    df["bid_depth"] = sum(df[f"bid{lvl}_sz"] for lvl in range(1, 6))
    df["ask_depth"] = sum(df[f"ask{lvl}_sz"] for lvl in range(1, 6))
    df["depth_imbalance"] = (df["bid_depth"] - df["ask_depth"]) / (df["bid_depth"] + df["ask_depth"] + 1e-9)
    df["microprice"] = (df["ask1"] * df["bid1_sz"] + df["bid1"] * df["ask1_sz"]) / (
                df["bid1_sz"] + df["ask1_sz"] + 1e-9)
    df["mp_minus_mid"] = df["microprice"] - df["mid"]
    df["mid_ret"] = df["mid"].pct_change()
    df["mid_ret_1s_proxy"] = df["mid"].diff()
    df["mid_vol_roll"] = df["mid_ret"].rolling(200).std(ddof=0)

    side = df["side"].astype(str).str.upper()
    df["trade_sign"] = np.where(side.str.contains("B"), 1,
                                np.where(side.str.contains("S"), -1, 0))
    df["signed_vol"] = df["trade_sign"] * df["trade_vol"].fillna(0)

    bpx, apx = df["bid1"], df["ask1"]
    bsz, asz = df["bid1_sz"].fillna(0), df["ask1_sz"].fillna(0)
    bpx_prev, apx_prev = bpx.shift(1), apx.shift(1)
    bsz_prev, asz_prev = bsz.shift(1), asz.shift(1)

    bid_contrib = np.where(bpx > bpx_prev, bsz,
                           np.where(bpx < bpx_prev, -bsz_prev, bsz - bsz_prev))
    ask_contrib = np.where(apx < apx_prev, asz,
                           np.where(apx > apx_prev, -asz_prev, asz - asz_prev))

    df["ofi_l1"] = bid_contrib - ask_contrib
    df["ofi_l1_roll"] = pd.Series(df["ofi_l1"]).rolling(200).sum()
    return df
