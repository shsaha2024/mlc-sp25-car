import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import os

# Standard library
import warnings
import ast
from pathlib import Path
from datetime import timedelta
from abc import ABC, abstractmethod
from collections import Counter

# Third-party libraries
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, skew, mannwhitneyu, kurtosis, entropy
from joblib import Parallel, delayed
import xgboost as xgb

# scikit-learn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    GridSearchCV,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    make_scorer
)
# Suppress warnings
warnings.filterwarnings("ignore")
import sys
init_dir = os.getcwd()


def c03_predictions(X_test):

    
    preds = np.zeros(X_test.shape[0])
    for fold in range(5):
        model = xgb.XGBClassifier()
        model.load_model(f"Final_workflow/c03_models/c03_model_{fold}.json")
        preds += model.predict_proba(X_test)[:, 1]
    
    preds /= 5  # average the predictions
    return preds

def features_c03(df_train, raw_consumer_file):
    def compute_trend(series):
        if not isinstance(series, list):
            return 0  # fallback if data malformed
    
        series = np.array(series)
        nonzero_idx = np.nonzero(series)[0]
        nonzero_vals = series[nonzero_idx]
    
        if len(nonzero_vals) <= 2:
            return 0
    
        X = nonzero_idx.reshape(-1, 1)  # time indices
        y = nonzero_vals
        model = LinearRegression().fit(X, y)
        return model.coef_[0]  # slope
    
    def process_user_parallel(user_id, group, ranges=24*7, num_categories=36):
        start = group['converted_date'].min()
        end = group['converted_date'].max()
        
        segments = []
        for x in range(int(start), int(end), ranges):
            mask = group[(group['converted_date'] >= x) & (group['converted_date'] < x + ranges)]
            if len(mask) == 0 or (mask['converted_date'].min() + ranges >  end):
                continue
    
            data = {
                'net_week_series': mask['amount'].sum(),
                'count_week_series': mask['amount'].count(),
                'inflow_week_series': mask[mask['amount'] > 0]['amount'].sum(),
                'outflow_week_series': mask[mask['amount'] < 0]['amount'].sum(),
            }
            for i in range(num_categories):
                data[f"week_series_category_{i}"] = mask[mask['category'] == i]['amount'].sum()
            segments.append(data)
    
        if len(segments) == 0:
            return user_id, pd.Series({
                key: [0] 
                for key in ['net_week_series', 'count_week_series', 'inflow_week_series', 'outflow_week_series'] +
                            [f"week_series_category_{i}" for i in range(num_categories)]
            })
    
        segments = segments[::-1]
        output = {key: [seg[key] for seg in segments] for key in segments[0].keys()}
        return user_id, pd.Series(output)
    
    
    def gini(array):
        array = np.sort(np.abs(array))
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array) + 1e-9)
    
    def compute_trend_2(x, y):
        """
        x: datetime-like index (e.g., DatetimeIndex)
        y: numeric values (e.g., average spend over time)
        """
        if len(x) < 2:
            return 0
        # Convert datetime index to days since min date
        x_days = (x - x.min()).days  # x must be a DatetimeIndex
        x_arr = np.array(x_days).reshape(-1, 1)
        y_arr = np.array(y)
        try:
            model = LinearRegression().fit(x_arr, y_arr)
            return model.coef_[0]
        except:
            return 0
    
    def compute_user_statistics(df_txn, suffix):
        df_txn['posted_date'] = pd.to_datetime(df_txn['posted_date'])
        df_txn['log_amount'] = np.log1p(np.abs(df_txn['amount']))
        df_txn['is_positive'] = df_txn['amount'] > 0
        df_txn['year_month'] = df_txn['posted_date'].dt.to_period('M')
        df_txn['week'] = df_txn['posted_date'].dt.isocalendar().week
        df_txn['dayofweek'] = df_txn['posted_date'].dt.dayofweek
    
        features = []
    
        for user_id, group in df_txn.groupby("masked_consumer_id"):
            amounts = group["amount"]
            dates = group["posted_date"].sort_values()
            gaps = dates.diff().dt.total_seconds().dropna() / (60*60*24)
            cat_counts = group['category'].value_counts(normalize=True)
    
            # Time grouping
            weekly = group.groupby(group['posted_date'].dt.to_period('W'))['amount']
            monthly = group.groupby('year_month')['amount']
            weekday = group.groupby('dayofweek')['amount']
    
            features.append({
                f'masked_consumer_id': user_id,
                f'amount_mean{suffix}': amounts.mean(),
                f'amount_median{suffix}': amounts.median(),
                f'amount_std{suffix}': amounts.std(),
                f'amount_var{suffix}': amounts.var(),
                f'amount_min{suffix}': amounts.min(),
                f'amount_max{suffix}': amounts.max(),
                f'amount_skew{suffix}': skew(amounts),
                f'amount_kurtosis{suffix}': kurtosis(amounts),
                f'amount_cv{suffix}': amounts.std() / (amounts.mean() + 1e-6),
                f'amount_gini{suffix}': gini(amounts.values),
                f'log_amount_mean{suffix}': group['log_amount'].mean(),
                f'txn_count{suffix}': len(group),
                f'txn_positive_ratio{suffix}': group['is_positive'].mean(),
                f'txn_duration_days{suffix}': (dates.max() - dates.min()).days + 1,
                f'txn_active_days{suffix}': dates.nunique(),
                f'txn_active_day_pct{suffix}': dates.nunique() / ((dates.max() - dates.min()).days + 1 + 1e-6),
                f'txn_gap_iqr{suffix}': np.subtract(*np.percentile(gaps, [75, 25])) if len(gaps) > 1 else 0,
                f'txn_burstiness{suffix}': gaps.std() if len(gaps) > 1 else 0,
                f'cat_count{suffix}': group['category'].nunique(),
                f'cat_concentration{suffix}': cat_counts.iloc[0] if len(cat_counts) > 0 else 0,
                f'net_spending_ratio{suffix}': amounts[amounts > 0].sum() / (amounts.abs().sum() + 1e-6),
                f'high_to_low_spend_ratio{suffix}': amounts.quantile(0.9) / (amounts.quantile(0.1) + 1e-6) if amounts.quantile(0.1) != 0 else 0,
                f'outlier_count{suffix}': ((amounts - amounts.mean()).abs() > 3 * amounts.std()).sum(),
                f'low_dollar_txn_pct{suffix}': (amounts.abs() < 5).mean(),
    
                # Temporal Features
                f'weekly_avg_spend{suffix}': weekly.mean().mean(),
                f'weekly_txn_freq{suffix}': weekly.count().mean(),
                f'weekly_spend_slope{suffix}': compute_trend_2(weekly.mean().index.to_timestamp(), weekly.mean()),
                f'monthly_avg_spend{suffix}': monthly.mean().mean(),
                f'monthly_txn_freq{suffix}': monthly.count().mean(),
                f'monthly_spend_slope{suffix}': compute_trend_2(monthly.mean().index.to_timestamp(), monthly.mean()),
                f'weekday_spend_std{suffix}': weekday.mean().std(),
                f'weekend_spend_pct{suffix}': group[group['dayofweek'] >= 5]['amount'].sum() / (amounts.sum() + 1e-6),
            })
    
        df_features = pd.DataFrame(features)
        return df_features
    
    #percentage by group (abs, total, pos only, neg only), ranking categories
    def compute_category_spending_distribution(df_txn):
        """
        Compute category-level spending distributions and test for significant differences
        between target groups (FPF_TARGET = 0 and 1).
    
        Returns:
            df_user_cat_stats: DataFrame with one row per user and percentage/rank features
            results_df: Mann-Whitney U test results with p-values for significance
        """
        df_txn['posted_date'] = pd.to_datetime(df_txn['posted_date'])
        df_txn['amount_abs'] = df_txn['amount'].abs()
        df_txn['amount_pos'] = df_txn['amount'].clip(lower=0)
        df_txn['amount_neg'] = df_txn['amount'].clip(upper=0).abs()
    
        user_cat_group = df_txn.groupby(['masked_consumer_id', 'category']).agg(
            amount_total=('amount', 'sum'),
            amount_abs=('amount_abs', 'sum'),
            amount_pos=('amount_pos', 'sum'),
            amount_neg=('amount_neg', 'sum'),
        ).reset_index()
    
        # Calculate total per user for normalizing
        user_totals = user_cat_group.groupby('masked_consumer_id').agg(
            total_amount_total=('amount_total', 'sum'),
            total_amount_abs=('amount_abs', 'sum'),
            total_amount_pos=('amount_pos', 'sum'),
            total_amount_neg=('amount_neg', 'sum'),
        ).reset_index()
    
        # Merge totals back into category group
        user_cat_group = user_cat_group.merge(user_totals, on='masked_consumer_id', how='left')
    
        # Compute percentage spend per category
        for col in ['amount_total', 'amount_abs', 'amount_pos', 'amount_neg']:
            user_cat_group[f'pct_{col}'] = user_cat_group[col] / (user_cat_group[f'total_{col}'] + 1e-6)
    
        # Compute per-user category rankings (lower rank = higher spend)
        user_cat_group['rank_total'] = user_cat_group.groupby('masked_consumer_id')['amount_total'].rank(ascending=False)
        user_cat_group['rank_abs'] = user_cat_group.groupby('masked_consumer_id')['amount_abs'].rank(ascending=False)
        user_cat_group['rank_pos'] = user_cat_group.groupby('masked_consumer_id')['amount_pos'].rank(ascending=False)
        user_cat_group['rank_neg'] = user_cat_group.groupby('masked_consumer_id')['amount_neg'].rank(ascending=False)
    
        # Pivot to user-level wide format
        pivot_cols = ['pct_amount_total', 'pct_amount_abs', 'pct_amount_pos', 'pct_amount_neg',
                    'rank_total', 'rank_abs', 'rank_pos', 'rank_neg']
        df_user_cat_stats = user_cat_group.pivot(index='masked_consumer_id', columns='category', values=pivot_cols)
        df_user_cat_stats.columns = ['{}_{}'.format(stat, cat) for stat, cat in df_user_cat_stats.columns]
        df_user_cat_stats = df_user_cat_stats.reset_index()

        # Attach target column
        target_map = df_txn[['masked_consumer_id']].drop_duplicates()
        #^ , 'FPF_TARGET'?

        df_user_cat_stats = df_user_cat_stats.merge(target_map, on='masked_consumer_id', how='left')
    
        return df_user_cat_stats
    
    def extract_fft_features(ts, i, detrend=True, low_freq_cut=0.05, high_freq_cut=0.25):
        ts = ts.asfreq('D').fillna(0)
        if detrend:
            ts = ts - ts.mean()
    
        fft_vals = np.fft.fft(ts)
        fft_freqs = np.fft.fftfreq(len(ts), d=1)
    
        pos_mask = fft_freqs > 0
        freqs = fft_freqs[pos_mask]
        mags = np.abs(fft_vals)[pos_mask]
        power = mags**2
        power_norm = power / power.sum()
    
        dominant_idx = np.argmax(power)
        dominant_freq = freqs[dominant_idx]
        dominant_power = power[dominant_idx]
        spectral_entropy = entropy(power_norm)
    
        low_power = power[(freqs < low_freq_cut)].sum()
        high_power = power[(freqs > high_freq_cut)].sum()
        power_ratio = low_power / (high_power + 1e-6)
    
        return {
            f"dominant_freq{i}": dominant_freq,
            f"dominant_power{i}": dominant_power,
            f"spectral_entropy{i}": spectral_entropy,
            f"low_freq_power{i}": low_power,
            f"high_freq_power{i}": high_power,
            f"power_ratio{i}": power_ratio,
        }
    
    def process_consumer(consumer_df, full_df, i):
        cid = consumer_df['masked_consumer_id'].iloc[0]
        final_balance = full_df.loc[full_df['masked_consumer_id'] == cid, 'total_balance'].values[0]
    
        consumer_df = consumer_df.sort_values('posted_date')
        consumer_df['cumulative_cashflow'] = consumer_df['amount'].cumsum()
    
        calc_final = consumer_df['cumulative_cashflow'].iloc[-1]
        offset = final_balance - calc_final
        consumer_df['shifted_cashflow'] = consumer_df['cumulative_cashflow'] + offset
    
        consumer_df['posted_date'] = pd.to_datetime(consumer_df['posted_date'])
        ts = (
            consumer_df
            .set_index('posted_date')
            .resample('D')['amount']
            .sum()
            .ffill()
        )
    
        ts_weekly = ts.resample('W').sum()
    
        features = extract_fft_features(ts_weekly, i)
        features['masked_consumer_id'] = cid
        return features

    ######START OF CODE######
    df_train = df_train[df_train['masked_consumer_id'].str[2] == '3']
    #create transaction_grouped
    grouped = list(df_train.groupby('masked_consumer_id'))
    results = Parallel(n_jobs=-1)(delayed(process_user_parallel)(uid, grp) for uid, grp in grouped)
    transaction_grouped = pd.DataFrame({uid: out for uid, out in results}).T.reset_index().rename(columns={'index': 'masked_consumer_id'})
    for col in transaction_grouped.columns[1:]:
        transaction_grouped[col] = transaction_grouped[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Apply to each time series column
    trend_df = pd.DataFrame()
    trend_df['masked_consumer_id'] = transaction_grouped['masked_consumer_id']
    
    # All time-series columns
    series_cols = transaction_grouped.columns.drop('masked_consumer_id')
    
    for col in series_cols:
        trend_df[col + '_trend'] = transaction_grouped[col].apply(compute_trend)

    df_train = df_train.merge(trend_df, on = 'masked_consumer_id', how = 'left')
    
    build_df = df_train.groupby('masked_consumer_id').agg({"total_balance": "mean", "week_series_category_0_trend":"mean", 
                                                    "week_series_category_1_trend":"mean", "week_series_category_2_trend":"mean",
                                                    "week_series_category_3_trend":"mean", "week_series_category_4_trend":"mean",
                                                    "week_series_category_5_trend":"mean", "week_series_category_6_trend":"mean",
                                                    "week_series_category_7_trend":"mean", "week_series_category_8_trend":"mean",
                                                    "week_series_category_9_trend":"mean", "week_series_category_10_trend":"mean",
                                                    "week_series_category_11_trend":"mean", "week_series_category_12_trend":"mean",
                                                    "week_series_category_13_trend":"mean", "week_series_category_14_trend":"mean",
                                                    "week_series_category_15_trend":"mean", "week_series_category_16_trend":"mean",
                                                    "week_series_category_17_trend":"mean", "week_series_category_18_trend":"mean",
                                                    "week_series_category_19_trend":"mean", "week_series_category_20_trend":"mean",
                                                    "week_series_category_21_trend":"mean", "week_series_category_22_trend":"mean",
                                                    "week_series_category_23_trend": "mean", "week_series_category_24_trend":"mean",
                                                    "week_series_category_25_trend": "mean", "week_series_category_26_trend":"mean",
                                                    "week_series_category_27_trend": "mean", "week_series_category_28_trend":"mean",
                                                    "week_series_category_29_trend": "mean", "week_series_category_30_trend":"mean",
                                                    "week_series_category_31_trend": "mean", "week_series_category_32_trend":"mean",
                                                    "week_series_category_33_trend": "mean", "week_series_category_34_trend": "mean",
                                                    "week_series_category_35_trend": "mean",
                                                    "FPF_TARGET":"first"}).reset_index()

    #, "FPF_TARGET": 'first'?
    
    build_df= build_df.merge(compute_user_statistics(df_train[df_train['amount'] <= 0], '_ovr_neg'), on = 'masked_consumer_id', how = 'left')
    build_df= build_df.merge(compute_user_statistics(df_train[df_train['amount'] > 0], '_ovr_pos'), on = 'masked_consumer_id', how = 'left')
    
    ## CONDUCT BY CATEGORY, CATEGORY GROUP
    for i in range(36):
        if i == 31:
            continue
        if i in [0, 1, 2, 4, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 28, 35]:
            one = compute_user_statistics(df_train[df_train['category'] == i], f'_{i}')
            if len(one) > 0:
                build_df= build_df.merge(one, on = 'masked_consumer_id', how = 'left')
        if i in [1, 11, 12]:    
            two = compute_user_statistics(df_train[(df_train['category'] == i) & (df_train['amount'] <= 0)], f'_{i}_neg')
            if len(two) > 0:
                build_df= build_df.merge(two, on = 'masked_consumer_id', how = 'left')
        if i in [0, 3, 11]:    
            three = compute_user_statistics(df_train[(df_train['category'] == i) & (df_train['amount'] > 0)], f'_{i}_pos')
            if len(three) > 0:
                build_df= build_df.merge(three, on = 'masked_consumer_id', how = 'left')
    
    ct = 0
    for group in [[32, 34, 24, 31, 13],[14, 18],[27, 13],[17, 36],[21, 20],[33, 29, 28], [24, 32, 34, 36, 29, 21, 27, 22, 31, 26],[14, 18, 28, 20, 30, 16, 19, 23, 25, 15],[25, 23, 10, 35, 36, 26, 12],[3, 5, 2, 6, 9, 8, 7, 11, 13, 0, 1]]:
        ct += 1
        if ct in [1, 2, 3, 6, 7, 8, 9, 10]:
            one = compute_user_statistics(df_train[df_train['category'].isin(group)], f'_group_{ct}')
            if len(one) > 0:
                build_df= build_df.merge(one, on = 'masked_consumer_id', how = 'left')
        if ct in [3, 9, 10]:
            two = compute_user_statistics(df_train[(df_train['category'].isin(group)) & (df_train['amount'] <= 0)], f'_group_{ct}_neg')
            if len(two) > 0:
                build_df= build_df.merge(two, on = 'masked_consumer_id', how = 'left')
        if ct in [9,10]:
            three = compute_user_statistics(df_train[(df_train['category'].isin(group)) & (df_train['amount'] > 0)], f'_group_{ct}_pos')
            if len(three) > 0:
                build_df= build_df.merge(three, on = 'masked_consumer_id', how = 'left')

    rank_cats = compute_category_spending_distribution(df_train)
    if 'FPF_TARGET' in rank_cats.columns:
        rank_cats.drop(columns = ['FPF_TARGET'])
    # Preview the filtered data
    end_df = build_df.merge(rank_cats, on = 'masked_consumer_id', how = 'left')

    # Main loop
    all_features = []
    
    consumer_df = pd.read_parquet(raw_consumer_file)
    X_consumer = consumer_df[consumer_df['masked_consumer_id'].str[2] == '3']
    
    
    for cid, group in df_train.groupby('masked_consumer_id'):
        try:
            features = process_consumer(group.copy(), consumer_df, 'ovr')
            all_features.append(features)
        except Exception as e:
            continue
            
    fft_features_df = pd.DataFrame(all_features)
    
    
    # Step 5: Merge transaction features into consumer-level data
    if "FPF_TARGET" in end_df.columns:
        end_df = end_df.drop(columns='FPF_TARGET')
    data = X_consumer.merge(end_df, on='masked_consumer_id', how='left')
    X_final = data.merge(fft_features_df, on='masked_consumer_id', how='left')
    for i in range(36):
        if i in [1, 3, 11, 13, 15, 16, 17, 22, 23, 24, 26]:
            all_features = []
            curr_df = df_train[df_train['category'] == i]
            for cid, group in curr_df.groupby('masked_consumer_id'):
                try:
                    features = process_consumer(group.copy(), consumer_df, i)
                    all_features.append(features)
                except Exception as e:
                    continue
            X_final = X_final.merge(pd.DataFrame(all_features), on = 'masked_consumer_id', how = 'left')
        
    
    X_final = X_final.fillna(0)

    cols = ['masked_consumer_id', 'high_freq_powerovr', 'amount_mean_ovr_pos', 'low_freq_powerovr', 'dominant_power26','weekly_spend_slope_group_9_pos',
    'log_amount_mean_group_9_neg','pct_amount_abs_13.0','weekly_txn_freq_11_neg','weekly_txn_freq_2','weekend_spend_pct_group_10_pos',
    'weekend_spend_pct_1','weekday_spend_std_group_6','amount_max_15','pct_amount_pos_0.0','power_ratio16','weekend_spend_pct_15',
    'txn_count_group_10_pos','weekday_spend_std_group_10','high_freq_power3','pct_amount_abs_26.0','amount_skew_6','amount_median_1',
    'log_amount_mean_11_pos','amount_skew_5','txn_active_day_pct_group_9_pos','high_to_low_spend_ratio_group_8','amount_std_4','amount_gini_1_neg',
    'amount_gini_7','log_amount_mean_23','monthly_txn_freq_19','pct_amount_total_10.0','monthly_avg_spend_15','amount_kurtosis_group_2',
    'weekly_txn_freq_18','outlier_count_ovr_neg','pct_amount_total_20.0','txn_active_day_pct_13','txn_active_day_pct_26','weekend_spend_pct_group_7',
    'log_amount_mean_group_3_neg','cat_concentration_group_8','monthly_spend_slope_18','txn_burstiness_19','log_amount_mean_28','power_ratio13',
    'monthly_avg_spend_group_3_neg','rank_total_10.0','outlier_count_group_8','high_to_low_spend_ratio_group_9','week_series_category_25_trend','dominant_freq22',
    'amount_median_6','amount_min_2','txn_positive_ratio_1','amount_min_4','amount_max_group_2','rank_total_26.0','outlier_count_4',
    'rank_pos_3.0','total_balance_x','low_dollar_txn_pct_12_neg','log_amount_mean_21','power_ratio24','amount_cv_group_2','monthly_avg_spend_20',
    'pct_amount_total_3.0','weekly_spend_slope_26','log_amount_mean_14','amount_cv_group_10_neg','monthly_spend_slope_0_pos','weekly_avg_spend_16',
    'weekday_spend_std_35','txn_gap_iqr_22','monthly_avg_spend_11_pos','high_to_low_spend_ratio_22','amount_skew_group_1','amount_gini_18','high_freq_power1',
    'week_series_category_13_trend','amount_kurtosis_17','amount_max_22','amount_skew_group_3','net_spending_ratio_1','amount_median_15','amount_std_19',
    'txn_active_day_pct_group_10','txn_gap_iqr_20','power_ratio17','week_series_category_6_trend','txn_burstiness_13','monthly_txn_freq_18','power_ratio1',
    'amount_cv_1_neg','log_amount_mean_18','txn_duration_days_2','amount_std_group_10','weekday_spend_std_group_9_neg','cat_concentration_group_2','amount_skew_7',
    'amount_kurtosis_4','high_freq_power15','power_ratio11','monthly_txn_freq_group_9_neg','dominant_powerovr','high_to_low_spend_ratio_7','amount_mean_12_neg','week_series_category_2_trend',
    'monthly_txn_freq_13','pct_amount_total_26.0','monthly_spend_slope_group_8','dominant_freq24','week_series_category_20_trend','rank_neg_17.0',
    'monthly_spend_slope_26','weekly_txn_freq_group_6','dominant_power23','weekend_spend_pct_group_10_neg','cat_concentration_group_6','amount_std_3_pos',
    'dominant_power1','weekly_avg_spend_0','monthly_txn_freq_group_1','amount_min_18','week_series_category_23_trend']
    
    X = X_final[cols]
    return X
