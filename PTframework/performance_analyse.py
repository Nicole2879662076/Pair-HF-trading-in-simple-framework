import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os
import re

def plot_strategy_analysis(df: pd.DataFrame, pos: pd.Series, trade_pnl: np.ndarray):
    """
    绘制高频配对交易的策略分析图表，用于报告展示。
    包含三个子图：
    1. Z-Score 及买卖信号点
    2. OFI 差值及动力确认阈值
    3. 逐笔交易的累计 PnL 曲线
    """
    # 提取信号触发的索引点
    pos_shift = pos.shift(1).fillna(0)
    long_entries = df.index[(pos == 1) & (pos_shift == 0)]
    short_entries = df.index[(pos == -1) & (pos_shift == 0)]
    exits = df.index[(pos == 0) & (pos_shift != 0)]

    # 创建画布
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1, 1]})
    fig.suptitle('Microstructure-Driven Pair Trading Strategy Analysis', fontsize=16, fontweight='bold')

    # ================= 1. 绘制 Z-Score 与 交易信号 =================
    ax1.plot(df.index, df['spread_z'], label='Microprice Spread Z-Score', color='royalblue', alpha=0.7, linewidth=1)
    ax1.axhline(1.5, color='red', linestyle='--', alpha=0.6, label='Short Entry Threshold (+1.5)')
    ax1.axhline(-1.5, color='green', linestyle='--', alpha=0.6, label='Long Entry Threshold (-1.5)')
    ax1.axhline(0, color='black', linestyle=':', alpha=0.6, label='Mean Reversion (0.0)')

    # 标记买卖点
    ax1.scatter(long_entries, df.loc[long_entries, 'spread_z'], marker='^', color='darkgreen', s=120,
                label='Buy Spread (Long A, Short B)', zorder=5)
    ax1.scatter(short_entries, df.loc[short_entries, 'spread_z'], marker='v', color='darkred', s=120,
                label='Sell Spread (Short A, Long B)', zorder=5)
    ax1.scatter(exits, df.loc[exits, 'spread_z'], marker='x', color='black', s=80, label='Exit (Reversion / Stop Loss)',
                zorder=5)

    ax1.set_title('1. Price Deviation: Spread Z-Score & Execution Points', fontsize=12)
    ax1.set_ylabel('Z-Score')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ================= 2. 绘制 OFI 动力确认 =================
    ax2.plot(df.index, df['ofi_diff'], label='OFI Difference (OFI_A - OFI_B)', color='purple', alpha=0.6, linewidth=1)
    ax2.axhline(5, color='orange', linestyle='--', alpha=0.8, label='OFI Long Momentum (> +5)')
    ax2.axhline(-5, color='orange', linestyle='--', alpha=0.8, label='OFI Short Momentum (< -5)')

    # 在 OFI 图上标记同样的入场点，以证明入场时 OFI 是达标的
    ax2.scatter(long_entries, df.loc[long_entries, 'ofi_diff'], marker='^', color='darkgreen', s=80, zorder=5)
    ax2.scatter(short_entries, df.loc[short_entries, 'ofi_diff'], marker='v', color='darkred', s=80, zorder=5)

    ax2.set_title('2. Momentum Confirmation: Order Flow Imbalance (OFI) Differential', fontsize=12)
    ax2.set_ylabel('OFI Diff')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ================= 3. 绘制 累计 PnL 曲线 =================
    if len(trade_pnl) > 0:
        cum_pnl = np.cumsum(trade_pnl)
        # 将盈亏点分成两类颜色：盈利为绿，亏损为红
        colors = ['green' if p > 0 else 'red' for p in trade_pnl]

        ax3.plot(range(1, len(cum_pnl) + 1), cum_pnl, color='teal', linewidth=2, label='Cumulative PnL')
        ax3.scatter(range(1, len(cum_pnl) + 1), cum_pnl, color=colors, s=50, zorder=5,
                    label='Trade Outcome (Green=Win, Red=Loss)')

        ax3.axhline(0, color='black', linewidth=1)
        ax3.set_title(f'3. Execution Result: Cumulative PnL Curve (Total Trades: {len(trade_pnl)})', fontsize=12)
        ax3.set_xlabel('Trade Sequence')
        ax3.set_ylabel('Cumulative PnL')
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Trades Executed', horizontalalignment='center', verticalalignment='center', fontsize=14)

    save_path = 'strategy_pnl.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 策略分析图已保存: {save_path}")
    plt.tight_layout()
    plt.show()
