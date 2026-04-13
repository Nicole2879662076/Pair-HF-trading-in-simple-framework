import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_processing import *
from pair_analyse import *
from performance_analyse import *


def align_two_files(stockA, stockB) -> pd.DataFrame:
    A_raw = load_all(stockA)
    B_raw = load_all(stockB)
    A_raw = clean_price_anomalies(A_raw)
    B_raw = clean_price_anomalies(B_raw)

    A = add_l2_and_orderflow_features(A_raw)
    B = add_l2_and_orderflow_features(B_raw)

    A = A[["timestamp", "mid", "bid1", "ask1", "depth_imbalance", "ofi_l1", "bid1_sz", "ask1_sz"]].rename(
        columns={"mid": "mid_A", "bid1": "bid_A", "ask1": "ask_A", "depth_imbalance": "imb_A", "ofi_l1": "ofi_A",
                 "bid1_sz": "bsz_A", "ask1_sz": "asz_A"})
    B = B[["timestamp", "mid", "bid1", "ask1", "depth_imbalance", "ofi_l1", "bid1_sz", "ask1_sz"]].rename(
        columns={"mid": "mid_B", "bid1": "bid_B", "ask1": "ask_B", "depth_imbalance": "imb_B", "ofi_l1": "ofi_B",
                 "bid1_sz": "bsz_B", "ask1_sz": "asz_B"})

    df = pd.merge_asof(A.sort_values("timestamp"), B.sort_values("timestamp"),
                       on="timestamp", direction="nearest")
    df = df.dropna().reset_index(drop=True)

    df['mp_A'] = (df['ask_A'] * df['bsz_A'] + df['bid_A'] * df['asz_A']) / (df['bsz_A'] + df['asz_A'] + 1e-9)
    df['mp_B'] = (df['ask_B'] * df['bsz_B'] + df['bid_B'] * df['asz_B']) / (df['bsz_B'] + df['asz_B'] + 1e-9)

    df["spread"] = np.log(df["mp_A"]) - np.log(df["mp_B"])
    lookback = 200
    df["spread_z"] = (df["spread"] - df["spread"].rolling(lookback).mean()) / (
                df["spread"].rolling(lookback).std() + 1e-9)
    df['ofi_diff'] = df['ofi_A'] - df['ofi_B']

    df['vol_10'] = df['mid_A'].pct_change().rolling(10).std()
    df['vol_50'] = df['mid_A'].pct_change().rolling(50).std()
    df['is_vol_high'] = df['vol_10'] > (df['vol_50'] * 1.2)

    return df.dropna().reset_index(drop=True)


def generate_pair_signals(df: pd.DataFrame) -> pd.Series:
    entry_z = 1.5
    exit_z = 0.0
    ofi_threshold = 5
    min_hold = 600
    max_hold = 3000
    stop_loss_pct = 0.002

    pos = pd.Series(0, index=df.index)
    curr_pos = 0
    ticks_held = 0

    z_scores = df['spread_z'].values
    ofi_diffs = df['ofi_diff'].values
    ask_A = df['ask_A'].values
    bid_A = df['bid_A'].values
    ask_B = df['ask_B'].values
    bid_B = df['bid_B'].values

    for i in range(len(df)):
        z = z_scores[i]
        o_diff = ofi_diffs[i]

        if curr_pos == 0:
            if z > entry_z and o_diff < -ofi_threshold:
                curr_pos = -1
                ticks_held = 0
                entry_price_A = bid_A[i]
                entry_price_B = ask_B[i]
            elif z < -entry_z and o_diff > ofi_threshold:
                curr_pos = 1
                ticks_held = 0
                entry_price_A = ask_A[i]
                entry_price_B = bid_B[i]
        else:
            ticks_held += 1
            current_pnl_pct = 0.0
            if curr_pos == -1:
                pnl_A = (entry_price_A - ask_A[i]) / entry_price_A
                pnl_B = (bid_B[i] - entry_price_B) / entry_price_B
                current_pnl_pct = pnl_A + pnl_B
            elif curr_pos == 1:
                pnl_A = (bid_A[i] - entry_price_A) / entry_price_A
                pnl_B = (entry_price_B - ask_B[i]) / entry_price_B
                current_pnl_pct = pnl_A + pnl_B

            if ticks_held >= min_hold:
                if abs(z) <= exit_z:
                    curr_pos = 0
                elif ticks_held >= max_hold:
                    curr_pos = 0
                elif current_pnl_pct <= -stop_loss_pct:
                    curr_pos = 0
                elif abs(z) > 5.0:
                    curr_pos = 0

        pos[i] = curr_pos
    return pos


def backtest_pair_cross_spread(df: pd.DataFrame, spread_pos: pd.Series):
    df = df.reset_index(drop=True)
    spread_pos = spread_pos.reset_index(drop=True)

    trade_pnl_list = []
    pos = 0
    entry_A = entry_B = 0.0
    cumulative_pnl = [0]
    cumulative_value = [1.0]
    trade_durations = []
    entry_times = []
    entry_prices = []

    for i in range(len(df)):
        desired = int(spread_pos[i])
        if desired == pos:
            continue

        if pos != 0:
            if pos == 1:
                pnl_A = df.at[i, "trade_bid_A"] - entry_A
                pnl_B = entry_B - df.at[i, "trade_ask_B"]
            else:
                pnl_A = entry_A - df.at[i, "trade_ask_A"]
                pnl_B = df.at[i, "trade_bid_B"] - entry_B

            trade_pnl_current = pnl_A + pnl_B
            trade_pnl_list.append(trade_pnl_current)

            if entry_times:
                duration = i - entry_times[-1]
                trade_durations.append(duration)

            cumulative_pnl.append(cumulative_pnl[-1] + trade_pnl_current)
            cumulative_value.append(1.0 + cumulative_pnl[-1])

        if desired != 0:
            if desired == 1:
                entry_A = df.at[i, "trade_ask_A"]
                entry_B = df.at[i, "trade_bid_B"]
            else:
                entry_A = df.at[i, "trade_bid_A"]
                entry_B = df.at[i, "trade_ask_B"]

            entry_times.append(i)
            entry_prices.append(entry_A - entry_B)

        pos = desired

    trade_pnl = np.array(trade_pnl_list) if trade_pnl_list else np.array([])

    if len(trade_pnl) > 0:
        total_pnl = np.sum(trade_pnl)
        num_trades = len(trade_pnl)
        win_rate = np.mean(trade_pnl > 0)
        avg_pnl = np.mean(trade_pnl)

        winning_trades = trade_pnl[trade_pnl > 0]
        losing_trades = trade_pnl[trade_pnl < 0]

        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0

        if len(losing_trades) > 0 and np.sum(losing_trades) != 0:
            profit_factor = abs(np.sum(winning_trades) / np.sum(losing_trades))
        else:
            profit_factor = np.inf if np.sum(winning_trades) > 0 else 0

        pnl_std = np.std(trade_pnl)
        cumulative_value_array = np.array(cumulative_value)
        running_max = np.maximum.accumulate(cumulative_value_array)
        drawdowns = (running_max - cumulative_value_array) / running_max
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        if pnl_std > 0:
            sharpe_ratio = avg_pnl / pnl_std
        else:
            sharpe_ratio = 0

        if pnl_std > 0:
            information_ratio = avg_pnl / pnl_std
        else:
            information_ratio = 0

        if len(losing_trades) > 1:
            downside_std = np.std(losing_trades)
        else:
            downside_std = 0

        if downside_std > 0:
            sortino_ratio = avg_pnl / downside_std
        else:
            sortino_ratio = np.inf if avg_pnl > 0 else 0

        if trade_durations:
            avg_duration = np.mean(trade_durations)
        else:
            avg_duration = 0

        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for pnl in trade_pnl:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0
    else:
        total_pnl = 0
        num_trades = 0
        win_rate = 0
        avg_pnl = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        pnl_std = 0
        max_drawdown = 0
        sharpe_ratio = 0
        information_ratio = 0
        sortino_ratio = 0
        avg_duration = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0

    return trade_pnl, {
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "pnl_std": pnl_std,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "information_ratio": information_ratio,
        "sortino_ratio": sortino_ratio,
        "avg_duration": avg_duration,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses
    }


def calculate_rolling_hedge_ratio_fast(pair_df, window_size=300):
    df = pair_df.copy()
    n = len(df)
    mid_A = df['mid_A'].values
    mid_B = df['mid_B'].values
    hedge_ratios = np.full(n, np.nan)

    for i in range(window_size, n):
        start = i - window_size
        end = i
        X_window = mid_B[start:end]
        y_window = mid_A[start:end]
        cov_xy = np.cov(X_window, y_window)[0, 1]
        var_x = np.var(X_window)

        if var_x > 1e-10:
            hedge_ratio = cov_xy / var_x
        else:
            hedge_ratio = np.nan

        hedge_ratios[i] = hedge_ratio

    df['hedge_ratio'] = hedge_ratios
    beta_pair_df = df.iloc[window_size:].reset_index(drop=True)
    return beta_pair_df


def calculate_trade_prices_with_hedge(beta_pair_df):
    df = beta_pair_df.copy()
    df['trade_bid_A'] = df['bid_A']
    df['trade_ask_A'] = df['ask_A']
    df['trade_bid_B'] = df['bid_B'] * df['hedge_ratio']
    df['trade_ask_B'] = df['ask_B'] * df['hedge_ratio']
    return df


