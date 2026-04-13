import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from data_processing import *
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.tsa.stattools import adfuller


def calculate_pair_correlation_simple(stockA, stockB):
    codeA = stockA.replace('.csv', '')
    codeB = stockB.replace('.csv', '')

    A = load_all(stockA)
    B = load_all(stockB)

    if A is None or B is None or len(A) == 0 or len(B) == 0:
        return None

    A = A[['timestamp', 'last']].copy() if 'last' in A.columns else None
    B = B[['timestamp', 'last']].copy() if 'last' in B.columns else None

    if A is None or B is None:
        return None

    A['timestamp'] = pd.to_datetime(A['timestamp'], errors='coerce')
    B['timestamp'] = pd.to_datetime(B['timestamp'], errors='coerce')

    A = A.rename(columns={'last': 'last_A'})
    B = B.rename(columns={'last': 'last_B'})

    A = A.sort_values('timestamp')
    B = B.sort_values('timestamp')

    merged = pd.merge_asof(A, B, on='timestamp', direction='nearest')

    if len(merged) < 10:
        return None

    merged['ret_A'] = merged['last_A'].pct_change()
    merged['ret_B'] = merged['last_B'].pct_change()

    data = merged[['ret_A', 'ret_B']].dropna()

    if len(data) < 10:
        return None

    corr, p_value = pearsonr(data['ret_A'], data['ret_B'])

    return {
        'stockA': codeA,
        'stockB': codeB,
        'correlation': corr,
        'p_value': p_value,
        'n_obs': len(data)
    }


def calculate_correlation_matrix_simple(pool, output_dir='correlation_results'):
    os.makedirs(output_dir, exist_ok=True)

    valid_stocks = []
    for stock in pool:
        code = stock.replace('.csv', '')
        data = load_all(stock)
        if data is not None and 'last' in data.columns and len(data) > 0:
            valid_stocks.append((stock, code))

    print(f"Valid stocks: {len(valid_stocks)}/{len(pool)}")
    print(f"Valid stock codes: {[code for _, code in valid_stocks]}")

    n = len(valid_stocks)
    stock_names = [code for _, code in valid_stocks]

    corr_matrix = np.full((n, n), np.nan)
    results = []

    print("\n" + "=" * 50)
    print("Starting correlation calculation...")
    print("=" * 50)

    for i in range(n):
        for j in range(i, n):
            stockA, codeA = valid_stocks[i]
            stockB, codeB = valid_stocks[j]

            result = calculate_pair_correlation_simple(stockA, stockB)

            if result is not None:
                corr = result['correlation']
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                results.append(result)
                print(f"✅ {codeA} - {codeB}: {corr:.4f}")
            else:
                print(f"⚠️  {codeA} - {codeB}: Cannot calculate")

    df_corr = pd.DataFrame(corr_matrix, index=stock_names, columns=stock_names)
    df_corr.to_csv(f'{output_dir}/correlation_matrix.csv')

    return df_corr


def plot_simple_correlation_heatmap(corr_matrix, output_dir='correlation_results'):
    plt.figure(figsize=(16, 16))
    colors = ["gold", "white", "mediumpurple"]
    custom_cmap = LinearSegmentedColormap.from_list("yellow_white_purple", colors, N=256)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix,
                xticklabels=False,
                yticklabels=False,
                mask=mask,
                annot=False,
                fmt=".2f",
                cmap=custom_cmap,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8})

    plt.title("Stock Return Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Heatmap saved: {output_dir}/correlation_heatmap.png")

    plt.show()


def main_simple_correlation_analysis(pool=None):
    if pool is None:
        pool = ['02513.csv', '01208.csv', '00941.csv', '00939.csv', '02888.csv', '06078.csv']

    print("🚀 Starting stock correlation analysis...")
    print(f"Stock pool: {pool}")

    df_corr = calculate_correlation_matrix_simple(pool, output_dir='correlation_results')
    plot_simple_correlation_heatmap(df_corr, output_dir='correlation_results')


def plot_simple_correlation_heatmap_from_csv(csv_path='correlation_results/correlation_matrix.csv',
                                             output_dir='correlation_results'):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"⚠️ File not found: {csv_path}")
        return

    corr_df = pd.read_csv(csv_path, dtype=str)
    stock_col = corr_df.columns[0]
    corr_df = corr_df.set_index(stock_col)
    corr_matrix = corr_df.apply(pd.to_numeric, errors='coerce')

    plt.figure(figsize=(16, 16))
    colors = ["gold", "white", "mediumpurple"]
    custom_cmap = LinearSegmentedColormap.from_list("yellow_white_purple", colors, N=256)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix,
                xticklabels=False,
                yticklabels=False,
                mask=mask,
                annot=False,
                fmt=".2f",
                cmap=custom_cmap,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8})

    plt.title("Stock Return Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Heatmap saved: {output_path}")

    plt.show()
    return output_path


def print_top_correlations_from_matrix(csv_path='correlation_results/correlation_matrix.csv', top_n=5):
    if not os.path.exists(csv_path):
        print(f"⚠️ File not found: {csv_path}")
        return

    corr_matrix = pd.read_csv(csv_path, index_col=0, dtype=str)
    corr_matrix = corr_matrix.apply(pd.to_numeric, errors='coerce')
    corr_matrix.index = corr_matrix.index.astype(str).str.zfill(5)
    corr_matrix.columns = corr_matrix.columns.astype(str).str.zfill(5)

    if len(corr_matrix) == 0:
        print("⚠️ Correlation matrix is empty")
        return

    correlations = []
    stocks = corr_matrix.index.tolist()

    for i in range(len(stocks)):
        for j in range(i + 1, len(stocks)):
            stockA = stocks[i]
            stockB = stocks[j]
            corr = corr_matrix.iloc[i, j]

            if not pd.isna(corr):
                correlations.append({
                    'stockA': stockA,
                    'stockB': stockB,
                    'correlation': corr,
                    'abs_corr': abs(corr)
                })

    if len(correlations) == 0:
        print("⚠️ No valid correlation data")
        return

    df = pd.DataFrame(correlations)

    print("\n" + "=" * 50)
    print(f"TOP {top_n} Highest Positive Correlations")
    print("=" * 50)
    pos_top = df[df['correlation'] > 0].nlargest(top_n, 'correlation')
    for i, (_, row) in enumerate(pos_top.iterrows(), 1):
        corr = row['correlation']
        print(f"{i:2d}. {row['stockA']} - {row['stockB']}: +{corr:.4f}")

    print("\n" + "=" * 50)
    print(f"TOP {top_n} Highest Negative Correlations")
    print("=" * 50)
    neg_top = df[df['correlation'] < 0].nsmallest(top_n, 'correlation')
    for i, (_, row) in enumerate(neg_top.iterrows(), 1):
        corr = row['correlation']
        print(f"{i:2d}. {row['stockA']} - {row['stockB']}: {corr:.4f}")

    print("\n" + "=" * 50)
    print(f"TOP {top_n} Weakest Correlations")
    print("=" * 50)
    weak_top = df.nsmallest(top_n, 'abs_corr')
    for i, (_, row) in enumerate(weak_top.iterrows(), 1):
        corr = row['correlation']
        sign = "+" if corr >= 0 else ""
        print(f"{i:2d}. {row['stockA']} - {row['stockB']}: {sign}{corr:.4f}")

    print("=" * 50)


def plot_pair_mid_price(pair_df, stockA_id, stockB_id, output_dir='plots', every_n_ticks=7200, figsize=(20, 8)):
    os.makedirs(output_dir, exist_ok=True)

    df_plot = pair_df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df_plot['timestamp']):
        df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'], errors='coerce')

    df_plot = df_plot.sort_values('timestamp')
    df_plot = df_plot.reset_index(drop=True)

    df_plot['timestamp_str'] = df_plot['timestamp'].dt.strftime('%m-%d %H:%M:%S')

    trading_days = df_plot['timestamp'].dt.date.nunique()

    plt.figure(figsize=figsize)

    plt.plot(df_plot['timestamp_str'], df_plot['mid_A'],
             linewidth=0.8, color='blue', alpha=0.8, label=f'{stockA_id} mid')

    plt.plot(df_plot['timestamp_str'], df_plot['mid_B'],
             linewidth=0.8, color='red', alpha=0.8, label=f'{stockB_id} mid')

    n = len(df_plot)
    if n > every_n_ticks * 2:
        indices = np.arange(0, n, every_n_ticks)
        indices = indices[indices < n]
        tick_labels = df_plot['timestamp_str'].iloc[indices].tolist()
        plt.xticks(indices, tick_labels, rotation=45, ha='right', fontsize=8)
    else:
        plt.xticks(rotation=45, ha='right')

    if n > 0:
        date_changes = df_plot['timestamp'].dt.date.diff() != pd.Timedelta(0)
        date_change_indices = df_plot[date_changes].index.tolist()

        for idx in date_change_indices[1:]:
            plt.axvline(x=idx, color='gray', alpha=0.3, linestyle=':',
                        linewidth=1, label='Day Change' if idx == date_change_indices[1] else "")

    plt.xlabel('Time (MM-DD HH:MM:SS)', fontsize=12)
    plt.ylabel('Mid Price', fontsize=12)
    plt.title(f'Mid Price Comparison: {stockA_id} vs {stockB_id} ({trading_days} trading days)',
              fontsize=14, fontweight='bold')

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='best')

    plt.tight_layout()

    filename = f"{stockA_id}_{stockB_id}_mid_comparison.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Price comparison chart saved: {filepath}")

    plt.show()

    return filepath


def plot_pair_mid_price_with_spread(pair_df, stockA_id, stockB_id, output_dir='plots', every_n_ticks=7200,
                                    figsize=(20, 10)):
    os.makedirs(output_dir, exist_ok=True)

    df_plot = pair_df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df_plot['timestamp']):
        df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'], errors='coerce')

    df_plot = df_plot.sort_values('timestamp')
    df_plot = df_plot.reset_index(drop=True)

    df_plot['timestamp_str'] = df_plot['timestamp'].dt.strftime('%m-%d %H:%M:%S')

    trading_days = df_plot['timestamp'].dt.date.nunique()

    spread_z = df_plot['spread_z'].dropna()
    if len(spread_z) > 0:
        spread_mean = spread_z.mean()
        spread_std = spread_z.std()
    else:
        spread_mean = 0
        spread_std = 1

    try:
        adf_result = adfuller(spread_z, autolag='AIC')
        adf_stat = adf_result[0]
        p_value = adf_result[1]
        is_stationary = p_value < 0.05
    except:
        adf_stat = np.nan
        p_value = np.nan
        is_stationary = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(df_plot['timestamp_str'], df_plot['mid_A'],
             linewidth=2, color='blue', alpha=0.8, label=f'{stockA_id} mid')
    ax1.plot(df_plot['timestamp_str'], df_plot['mid_B'],
             linewidth=2, color='red', alpha=0.8, label=f'{stockB_id} mid')

    ax1.set_ylabel('Mid Price', fontsize=12)
    ax1.set_title(f'Mid Price Comparison: {stockA_id} vs {stockB_id} ({trading_days} trading days)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, loc='best')

    ax2.plot(df_plot['timestamp_str'], df_plot['spread_z'],
             linewidth=0.8, color='green', alpha=0.8, label='spread_z')

    ax2.axhspan(spread_mean - spread_std, spread_mean + spread_std,
                alpha=0.5, color='gray', label='±1σ')
    ax2.axhspan(spread_mean - 2 * spread_std, spread_mean - spread_std,
                alpha=0.5, color='yellow')
    ax2.axhspan(spread_mean + spread_std, spread_mean + 2 * spread_std,
                alpha=0.5, color='yellow')

    ax2.axhline(y=0, color='black', linewidth=0.5, alpha=0.5, linestyle='-')

    ax2.axhline(y=spread_mean, color='red', linewidth=1, alpha=0.5,
                linestyle='--', label=f'mean={spread_mean:.2f}')

    ax2.set_xlabel('Time (MM-DD HH:MM:SS)', fontsize=12)
    ax2.set_ylabel('Spread Z-score', fontsize=12)

    adf_text = f'ADF={adf_stat:.2f}, p={p_value:.3f} ({"" if is_stationary else "non-"}stationary)'
    ax2.set_title(f'Spread Z-score | Mean={spread_mean:.2f}, Std={spread_std:.2f} | {adf_text}',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9, loc='best')

    n = len(df_plot)
    if n > every_n_ticks * 2:
        indices = np.arange(0, n, every_n_ticks)
        indices = indices[indices < n]
        tick_labels = df_plot['timestamp_str'].iloc[indices].tolist()
        ax2.set_xticks(indices)
        ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    else:
        ax2.tick_params(axis='x', rotation=45)

    if n > 0:
        date_changes = df_plot['timestamp'].dt.date.diff() != pd.Timedelta(0)
        date_change_indices = df_plot[date_changes].index.tolist()

        for idx in date_change_indices[1:]:
            ax1.axvline(x=idx, color='gray', alpha=0.3, linestyle=':', linewidth=1)
            ax2.axvline(x=idx, color='gray', alpha=0.3, linestyle=':', linewidth=1)

    plt.tight_layout()

    filename = f"{stockA_id}_{stockB_id}_mid_spread_comparison.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Price comparison chart saved: {filepath}")

    plt.show()

    return filepath


def remove_stock_from_correlation_matrix(csv_path='correlation_results/correlation_matrix.csv', stock_to_remove='2197'):
    if not os.path.exists(csv_path):
        print(f"⚠️ File not found: {csv_path}")
        return None

    corr_matrix = pd.read_csv(csv_path, dtype=str)
    stock_col = corr_matrix.columns[0]
    corr_matrix.set_index(stock_col, inplace=True)

    corr_matrix.index = corr_matrix.index.astype(str)
    corr_matrix.columns = corr_matrix.columns.astype(str)

    stock_to_remove = str(stock_to_remove)
    if stock_to_remove not in corr_matrix.index:
        print(f"⚠️ Stock {stock_to_remove} not in correlation matrix")
        return corr_matrix

    print(f"Removing stock {stock_to_remove}...")
    corr_matrix = corr_matrix.drop(index=stock_to_remove, columns=stock_to_remove)

    corr_matrix = corr_matrix.reset_index()
    corr_matrix = corr_matrix.rename(columns={'index': stock_col})

    corr_matrix.to_csv(csv_path, index=False)
    print(f"✅ Stock {stock_to_remove} removed from correlation matrix")

    return corr_matrix

# remove_stock_from_correlation_matrix('correlation_results/correlation_matrix.csv', '02197')
# lot_simple_correlation_heatmap_from_csv('correlation_results/correlation_matrix.csv', 'correlation_results')

# if __name__ == "__main__":
#     stock_pool = ['01208.csv', '00941.csv', '00939.csv', '02888.csv', '06078.csv', '01024.csv', '00763.csv',
#             '02331.csv', '01530.csv', '02600.csv', '00981.csv', '02382.csv', '01093.csv', '09926.csv', '00388.csv',
#             '02197.csv', '03993.csv', '01177.csv', '01729.csv', '01810.csv', '01801.csv', '09999.csv', '01929.csv',
#             '06869.csv', '06990.csv', '01378.csv', '01339.csv', '01877.csv', '02313.csv', '01347.csv', '09618.csv',
#             '02318.csv', '03690.csv', '03896.csv', '01398.csv', '06160.csv', '00883.csv', '01299.csv',
#             '02628.csv', '03330.csv', '06181.csv', '02171.csv', '02145.csv', '01211.csv', '02899.csv', '09988.csv',
#             '09985.csv', '09995.csv', '01815.csv', '00358.csv', '02015.csv', '02097.csv', '01258.csv', '02269.csv',
#             '09868.csv', '00700.csv', '09903.csv', '00998.csv', '02020.csv', '01772.csv', '09992.csv', '00005.csv',
#             '03692.csv', '01788.csv', '09991.csv', '00340.csv', '09898.csv', '02099.csv']
#
#     main_simple_correlation_analysis(stock_pool)
#     print_top_correlations_from_matrix('correlation_results/correlation_matrix.csv', top_n=10)
