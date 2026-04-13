from pair_trade_strategy import *

pair_df = align_two_files("02888.csv", "00005.csv")
pair_df = pair_df.sort_values('timestamp')
print(pair_df.columns)
# plot_pair_mid_price(pair_df, "02888", "00005")
# plot_pair_mid_price_with_spread(pair_df, "02888", "00005")
beta_df = calculate_rolling_hedge_ratio_fast(pair_df, window_size=600)
trade_beta_df  = calculate_trade_prices_with_hedge(beta_df)

pair_pos = generate_pair_signals(trade_beta_df)
pair_pnl, pair_stats = backtest_pair_cross_spread(trade_beta_df, pair_pos)
print("Pair stats:", pair_stats)

plot_strategy_analysis(trade_beta_df, pair_pos, pair_pnl)