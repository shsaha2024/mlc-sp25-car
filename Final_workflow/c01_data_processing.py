#!/usr/bin/env python3
# feature_pipeline_with_prefixes.py

"""
Feature engineering pipeline with distinct prefixes per block and simulations,
carrying evaluation_date into simulated rows to maintain dtype consistency.
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import entropy
from typing import Optional, Dict
from sklearn.linear_model import LinearRegression

# Number of simulations per positive consumer
N_SIMULATIONS = 20

def read_parquet_auto(path: str) -> pd.DataFrame:
    print(f"Loading parquet: {path}")
    for engine in (None, 'pyarrow', 'fastparquet'):
        try:
            df = pd.read_parquet(path, engine=engine) if engine else pd.read_parquet(path)
            print(f" -> loaded via {engine or 'default'} engine, shape={df.shape}")
            return df
        except ImportError:
            continue
        except Exception:
            continue
    print(f"ERROR: could not read {path}. Install pyarrow or fastparquet.")
    sys.exit(1)

def weekly_trimmed(ts: pd.Series) -> pd.Series:
    ts2 = ts.groupby(ts.index).sum().asfreq('D').fillna(0)
    if ts2.empty:
        return ts2
    first = ts2.index[0] + pd.DateOffset(days=(7 - ts2.index[0].weekday()) % 7)
    last  = ts2.index[-1] - pd.DateOffset(days=(ts2.index[-1].weekday() + 1) % 7)
    ts2 = ts2[(ts2.index >= first) & (ts2.index <= last)]
    return ts2.resample('W-MON').sum()

def extract_fft_features(ts: pd.Series,
                         detrend: bool=True,
                         low_freq_cut: float=0.05,
                         high_freq_cut: float=0.25) -> Optional[Dict[str,float]]:
    series = ts.asfreq('W-MON').fillna(0)
    if detrend:
        series = series - series.mean()
    vals    = series.values
    freqs   = np.fft.fftfreq(len(vals), d=1)
    fft_vals= np.fft.fft(vals)
    pos     = freqs > 0
    freqs   = freqs[pos]
    power   = np.abs(fft_vals[pos])**2
    total   = power.sum()
    if total == 0:
        return None
    p_norm  = power / total
    idx     = int(np.argmax(power))
    return {
        'fft_dominant_freq':    freqs[idx],
        'fft_dominant_power':   power[idx],
        'fft_spectral_entropy': entropy(p_norm),
        'fft_low_freq_power':   power[freqs < low_freq_cut].sum(),
        'fft_high_freq_power':  power[freqs > high_freq_cut].sum(),
        'fft_power_ratio':      power[freqs < low_freq_cut].sum() /
                                (power[freqs > high_freq_cut].sum() + 1e-6)
    }

def make_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a rich set of features per masked_consumer_id:
      - Basic aggregates (sum, mean, std, etc.)
      - Recent 30-day sums/counts
      - Monthly volatility and trend
      - Weekend vs weekday ratios
      - Transaction frequency metrics
      - Amount percentiles
      - Large transaction counts
      - Category-based encodings and spend/gap statistics
      - Timing gaps per category
    This version enforces that masked_consumer_id remains a column,
    coerces types, drops only invalid rows, and warns on empty steps.
    """
    tx = transactions.copy()

    # ── 1) Ensure ID column ────────────────────────────────────────────────────
    if 'masked_consumer_id' not in tx.columns:
        tx = tx.reset_index()
    tx['masked_consumer_id'] = tx['masked_consumer_id'].astype(str)

    # ── 2) Coerce types & drop invalid ───────────────────────────────────────
    if 'amount' not in tx.columns:
        raise KeyError("make_transaction_features: 'amount' column is required")
    tx['amount'] = pd.to_numeric(tx['amount'], errors='coerce')
    tx['posted_date'] = pd.to_datetime(tx.get('posted_date', None), errors='coerce')
    # evaluation_date may or may not exist
    if 'evaluation_date' in tx.columns:
        tx['evaluation_date'] = pd.to_datetime(tx['evaluation_date'], errors='coerce')

    tx = tx.dropna(subset=['masked_consumer_id', 'amount', 'posted_date'])
    if tx.empty:
        print("⚠️ make_transaction_features: after dropna, no transactions remain.")
        return pd.DataFrame(columns=['masked_consumer_id'])

    # ── 3) Basic aggregates ───────────────────────────────────────────────────
    tx['is_credit'] = tx['amount'] > 0
    tx['is_debit']  = tx['amount'] < 0

    agg = tx.groupby('masked_consumer_id').agg(
        total_amount      = ('amount','sum'),
        mean_amount       = ('amount','mean'),
        std_amount        = ('amount','std'),
        min_amount        = ('amount','min'),
        max_amount        = ('amount','max'),
        median_amount     = ('amount','median'),
        transaction_count = ('amount','count'),
        credit_sum        = ('is_credit', lambda x: tx.loc[x.index, 'amount'][x].sum()),
        debit_sum         = ('is_debit',  lambda x: abs(tx.loc[x.index, 'amount'][x].sum()))
    )
    if agg.empty:
        print("⚠️ make_transaction_features: basic agg returned empty DataFrame.")
    agg['credit_debit_ratio'] = (
        agg['credit_sum'] / agg['debit_sum'].replace(0, np.nan)
    ).fillna(0)

    # ── 4) Recent 30-day features ─────────────────────────────────────────────
    if 'evaluation_date' in tx.columns:
        tx['days_before_eval'] = (tx['evaluation_date'] - tx['posted_date']).dt.days
        recent_df = (
            tx[tx['days_before_eval'] <= 30]
            .groupby('masked_consumer_id')
            .agg(recent30_sum=('amount','sum'),
                 recent30_count=('amount','count'))
        )
    else:
        recent_df = pd.DataFrame()

    # ── 5) Monthly volatility ─────────────────────────────────────────────────
    tx['month'] = tx['posted_date'].dt.to_period('M')
    monthly_cashflow = (
        tx.groupby(['masked_consumer_id','month'])['amount']
          .sum().reset_index()
    )
    monthly_stats = (
        monthly_cashflow
        .groupby('masked_consumer_id')['amount']
        .agg(['mean','std'])
        .rename(columns={'mean':'monthly_mean','std':'monthly_std'})
    )
    if monthly_stats.empty:
        print("⚠️ make_transaction_features: monthly_stats empty.")
    monthly_stats['monthly_cv'] = (
        monthly_stats['monthly_std'] /
        monthly_stats['monthly_mean'].replace(0, np.nan)
    ).fillna(0)

    # ── 6) Weekend vs weekday spending ────────────────────────────────────────
    tx['weekday']    = tx['posted_date'].dt.weekday
    tx['is_weekend'] = tx['weekday'] >= 5
    weekend_spend = (
        tx[tx['amount'] > 0]
        .groupby('masked_consumer_id')
        .agg(
            weekday_spending=('amount', lambda x: x[~tx.loc[x.index,'is_weekend']].sum()),
            weekend_spending=('amount', lambda x: x[ tx.loc[x.index,'is_weekend'] ].sum())
        )
    )
    if weekend_spend.empty:
        print("⚠️ make_transaction_features: weekend_spend empty.")
    weekend_spend['weekend_ratio'] = (
        weekend_spend['weekend_spending'] /
        (weekend_spend['weekday_spending'] + 1e-6)
    )

    # ── 7) Transaction frequency ──────────────────────────────────────────────
    txn_freq = (
        tx.groupby('masked_consumer_id')
          .agg(
            unique_txn_days=('posted_date', lambda x: x.nunique()),
            txn_days_span   =('posted_date', lambda x: (x.max()-x.min()).days + 1)
          )
    )
    txn_freq['txn_per_day'] = (
        txn_freq['unique_txn_days'] /
        txn_freq['txn_days_span'].replace(0, 1)
    )

    # ── 8) Amount percentiles ────────────────────────────────────────────────
    percentiles = (
        tx.groupby('masked_consumer_id')['amount']
          .quantile([0.25, 0.75, 0.9]).unstack()
    )
    percentiles.columns = ['amount_25pct','amount_75pct','amount_90pct']

    # ── 9) Large transaction count ────────────────────────────────────────────
    tx['is_large'] = tx['amount'].abs() > 1000
    large_txns    = tx.groupby('masked_consumer_id')['is_large'].sum()

    # ── 10) Monthly spend trend ───────────────────────────────────────────────
    monthly_spend = monthly_cashflow.copy()
    if not monthly_spend.empty:
        monthly_spend['month'] = monthly_spend['month'].dt.to_timestamp()
        def compute_monthly_trend(df):
            if len(df) < 2:
                return 0.0
            df = df.sort_values('month')
            x = (df['month'] - df['month'].min()).dt.days.values.reshape(-1,1)
            y = df['amount'].values
            return float(LinearRegression().fit(x, y).coef_[0])
        monthly_trend = monthly_spend.groupby('masked_consumer_id').apply(compute_monthly_trend)
    else:
        monthly_trend = pd.Series(dtype=float)

    # ── 11) Category one-hot sums ─────────────────────────────────────────────
    cat_encode      = pd.get_dummies(tx, columns=['category'], prefix='cat', drop_first=False)
    cat_cols        = [c for c in cat_encode if c.startswith('cat_')]
    cat_encode_mean = cat_encode.groupby('masked_consumer_id')[cat_cols].sum()

    # ── 12) Category spend pivot ──────────────────────────────────────────────
    if 'category' in tx:
        cat_spending = (
            tx.pivot_table(
                index='masked_consumer_id',
                columns='category',
                values='amount',
                aggfunc='sum',
                fill_value=0
            )
            .rename(columns=lambda c: f'spend_cat_{int(c)}')
        )
    else:
        cat_spending = pd.DataFrame()

    # ── 13) Credit-debit gap per category ────────────────────────────────────
    def credit_debit_gap(df):
        pos = df[df['amount'] > 0].groupby('category')['amount'].sum()
        neg = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        return (pos - neg).fillna(0)
    gap_df = (
        tx.groupby('masked_consumer_id')
          .apply(credit_debit_gap)
          .unstack()
          .add_prefix('gap_cat_')
          .fillna(0)
    )

    # ── 14) Binary category usage ─────────────────────────────────────────────
    if 'category' in tx:
        usage = (
            pd.crosstab(tx['masked_consumer_id'], tx['category'])
              .astype(bool).astype(int)
              .rename(columns=lambda c: f'used_cat_{int(c)}')
        )
    else:
        usage = pd.DataFrame()

    # ── 15) Timing between transactions per category ─────────────────────────
    cat_days = tx.groupby(['masked_consumer_id','category'])['posted_date'].agg(['min','max','count'])
    cat_days['duration'] = (cat_days['max'] - cat_days['min']).dt.days
    cat_days['txn_gap']  = cat_days['duration'] / cat_days['count'].replace(0,1)
    duration_feat = (
        cat_days['txn_gap']
        .unstack(fill_value=0)
        .add_prefix('gap_days_cat_')
    )

    # ── 16) Merge everything ──────────────────────────────────────────────────
    parts = {
        'agg':                agg,
        'recent_df':          recent_df,
        'monthly_stats':      monthly_stats,
        'txn_freq':           txn_freq,
        'weekend_spend':      weekend_spend,
        'large_txns':         large_txns.to_frame('large_txns'),
        'percentiles':        percentiles,
        'monthly_trend':      monthly_trend.to_frame('monthly_trend'),
        'cat_encode_mean':    cat_encode_mean,
        'cat_spending':       cat_spending,
        'gap_df':             gap_df,
        'usage':              usage,
        'duration_feat':      duration_feat
    }

    # Convert any Series to DataFrame, then join
    for name, df_part in parts.items():
        if isinstance(df_part, pd.Series):
            parts[name] = df_part.to_frame(df_part.name or name)

    # Start with agg, then left-join all others
    features_df = parts['agg']
    for name, df_part in parts.items():
        if name == 'agg':
            continue
        features_df = features_df.join(df_part, how='left')

    features_df = features_df.fillna(0).reset_index()
    return features_df

def make_weekly_features(tx: pd.DataFrame,
                         top_categories: Optional[list]=None,
                         include_cv: bool=True,
                         include_trend: bool=True,
                         include_rolling: bool=True,
                         include_weekend: bool=True) -> pd.DataFrame:
    print("Computing weekly-based category features (week_ prefix)...")
    tx['posted_date'] = pd.to_datetime(tx['posted_date'])
    tx['week']        = tx['posted_date'].dt.to_period('W').apply(lambda r: r.start_time)

    if top_categories is None:
        top_categories = tx['category'].value_counts().nlargest(35).index.tolist()
    txf    = tx[tx['category'].isin(top_categories)].copy()
    weekly = txf.groupby(['masked_consumer_id','category','week'])['amount'].sum().reset_index()

    parts = []
    if include_cv:
        vol = weekly.groupby(['masked_consumer_id','category'])['amount'].agg(['mean','std'])
        vol['cv'] = (vol['std']/vol['mean'].replace(0,np.nan)).fillna(0)
        vol = vol.unstack(fill_value=0)
        vol.columns = [f'week_cv_cat_{cat}' for cat in vol.columns.get_level_values(1)]
        parts.append(vol)

    if include_trend:
        trends = (
            weekly.groupby(['masked_consumer_id','category'])
                  .apply(lambda df: LinearRegression()
                         .fit(
                             (df['week'] - df['week'].min()).dt.days.values.reshape(-1,1),
                             df['amount']
                         ).coef_[0] if len(df)>=2 else 0)
                  .unstack(fill_value=0)
        )
        trends.columns = [f'week_trend_cat_{cat}' for cat in trends.columns]
        parts.append(trends)

    if include_rolling:
        roll = weekly.copy()
        roll['rolling_mean']   = roll.groupby(['masked_consumer_id','category'])['amount']\
                                     .transform(lambda x: x.rolling(3, min_periods=1).mean())
        roll['weekly_change']  = roll.groupby(['masked_consumer_id','category'])['amount']\
                                     .pct_change().replace([np.inf,-np.inf],0).fillna(0)
        agg_roll = roll.groupby(['masked_consumer_id','category']).agg({
            'rolling_mean': 'mean',
            'weekly_change': ['mean','std']
        })
        agg_roll.columns = [f'week_roll_{stat}_cat_{cat}' for stat,cat in agg_roll.columns]
        agg_roll = agg_roll.unstack(fill_value=0)
        parts.append(agg_roll)

    if include_weekend:
        txf['is_weekend'] = txf['week'].dt.weekday >= 5
        weekend = txf.groupby(['masked_consumer_id','category'])\
                     .apply(lambda df: df.loc[df['is_weekend'],'amount'].sum() /
                                   (df.loc[~df['is_weekend'],'amount'].sum() + 1e-6)
                            ).unstack(fill_value=0)
        weekend.columns = [f'weekend_ratio_cat_{cat}' for cat in weekend.columns]
        parts.append(weekend)

    final = pd.concat(parts, axis=1).fillna(0)
    final.columns = ['_'.join(map(str,c)) if isinstance(c, tuple) else c for c in final.columns]
    print(f" -> {final.shape}")
    return final

def compute_fft_for_tx(tx_consumer: pd.DataFrame) -> Dict[str,float]:
    feats = {}
    for cat, sub in tx_consumer.groupby('category'):
        ts = weekly_trimmed(sub.set_index('posted_date')['amount'])
        if len(ts) < 6 or ts.sum() == 0:
            continue
        ff = extract_fft_features(ts)
        if ff:
            feats.update({f'{k}_cat{cat}': v for k,v in ff.items()})
    return feats

def main():
    consumer_file = '/Users/jasonc/Desktop/DSC_291/cashflow/consumer_data.parquet'
    tx_file       = '/Users/jasonc/Desktop/DSC_291/cashflow/transactions.parquet'
    out_file      = '/Users/jasonc/Desktop/DSC_291/merged_features.parquet'

    consumer = read_parquet_auto(consumer_file).set_index('masked_consumer_id')
    tx       = read_parquet_auto(tx_file)

    print("Merging evaluation_date into tx if present...")
    if 'evaluation_date' in consumer.columns:
        tx['posted_date'] = pd.to_datetime(tx['posted_date'])
        tx = tx.merge(
            consumer['evaluation_date'],
            left_on='masked_consumer_id',
            right_index=True
        )

    # Filter to C01 group
    consumer = consumer[consumer.index.str.startswith('C01')]
    tx       = tx[tx['masked_consumer_id'].isin(consumer.index)]
    print(f"Filtered: {consumer.shape[0]} consumers, {tx.shape[0]} transactions")

    # Real features
    agg_df  = make_transaction_features(tx).set_index('masked_consumer_id')
    week_df = make_weekly_features(tx)

    # FFT features
    print("Calculating FFT Features...")
    fft_rows = []
    for cid in tx['masked_consumer_id'].unique():
        feats = compute_fft_for_tx(tx[tx['masked_consumer_id']==cid])
        feats['masked_consumer_id'] = cid
        fft_rows.append(feats)
    fft_df = pd.DataFrame(fft_rows).set_index('masked_consumer_id').fillna(0)
    print(f"FFT features: {fft_df.shape}")

    # Merge real
    real = agg_df.join(week_df, how='left').join(fft_df, how='left').fillna(0)
    real['dataset_type'] = 'real'
    if 'FPF_TARGET' in consumer.columns:
        real['FPF_TARGET'] = consumer['FPF_TARGET']
    print(f"Real merged: {real.shape}")

    # ─── SIMULATION BLOCK ────────────────────────────────────────────────────────
    sim_list = []
    pos_ids = real.loc[real['FPF_TARGET'] == 1].index
    print(f"Simulating {N_SIMULATIONS} per {len(pos_ids)} positive IDs...")

    for cid in pos_ids:
        tx_cons = tx[tx['masked_consumer_id'] == cid]

        # Fix 1: require a minimum number of transactions
        if len(tx_cons) < 10:
            print(f"Skipping all sims for {cid}: only {len(tx_cons)} transactions.")
            continue

        # Fix 2: require some variation
        raw_std = tx_cons['amount'].std()
        if pd.isna(raw_std) or raw_std == 0:
            print(f"Skipping all sims for {cid}: no variation (std={raw_std}).")
            continue

        eval_date = consumer.at[cid, 'evaluation_date'] if 'evaluation_date' in consumer.columns else pd.NaT

        for sim in range(1, N_SIMULATIONS + 1):
            sim_tx = tx_cons.copy()
            if sim_tx.empty:
                print(f"  • Sim {sim} for {cid}: empty data → skip")
                continue

            # add noise
            noise_scale = sim_tx['amount'].std() * 0.1
            if pd.isna(noise_scale) or noise_scale == 0:
                print(f"  • Sim {sim} for {cid}: invalid noise_scale={noise_scale} → skip")
                continue
            sim_tx['amount'] += np.random.normal(0, noise_scale, size=len(sim_tx))

            # aggregate features
            agg_df = make_transaction_features(sim_tx).set_index('masked_consumer_id')
            if cid not in agg_df.index:
                print(f"  • Sim {sim}: agg missing → tx.shape={sim_tx.shape}, noise={noise_scale}")
                continue
            agg_feats = agg_df.loc[cid].to_dict()

            # weekly features
            week_df = make_weekly_features(sim_tx).set_index('masked_consumer_id')
            if cid not in week_df.index:
                print(f"  • Sim {sim}: weekly missing → week_idx={week_df.index.tolist()}")
                continue
            week_feats = week_df.loc[cid].to_dict()

            # FFT features (may return empty dict)
            fft_feats = compute_fft_for_tx(sim_tx)

            # build the row
            sim_list.append({
                'masked_consumer_id': f"{cid}_sim{sim}",
                'evaluation_date': eval_date,
                'FPF_TARGET': 1,
                'dataset_type': 'simulation',
                **agg_feats,
                **week_feats,
                **fft_feats
            })

    # finalize
    if sim_list:
        sim_df = pd.DataFrame(sim_list).set_index('masked_consumer_id').fillna(0)
        print(f"-> Simulated rows: {sim_df.shape}")
    else:
        sim_df = pd.DataFrame()
        print("-> No valid simulations created.")
    # ─────────────────────────────────────────────────────────────────────────────



    # Final concat & save
    real   = real.loc[:, ~real.columns.duplicated()]
    sim_df = sim_df.loc[:, ~sim_df.columns.duplicated()]
    final  = pd.concat([real, sim_df], axis=0).fillna(0)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    final.to_parquet(out_file)
    print(f"Saved final dataset: {final.shape} -> {out_file}")

if __name__ == '__main__':
    main()
