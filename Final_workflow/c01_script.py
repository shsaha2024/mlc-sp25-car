# feature_pipeline_with_prefixes.py
#!/usr/bin/env python3
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
    last = ts2.index[-1] - pd.DateOffset(days=(ts2.index[-1].weekday() + 1) % 7)
    ts2 = ts2[(ts2.index >= first) & (ts2.index <= last)]
    return ts2.resample('W-MON').sum()


def extract_fft_features(ts: pd.Series,
                         detrend: bool=True,
                         low_freq_cut: float=0.05,
                         high_freq_cut: float=0.25) -> Optional[Dict[str,float]]:
    series = ts.asfreq('W-MON').fillna(0)
    if detrend:
        series = series - series.mean()
    vals = series.values
    freqs = np.fft.fftfreq(len(vals), d=1)
    fft_vals = np.fft.fft(vals)
    pos = freqs > 0
    freqs, mags = freqs[pos], np.abs(fft_vals)[pos]
    power = mags**2
    total = power.sum()
    if total == 0:
        return None
    p_norm = power/total
    idx = int(np.argmax(power))
    return {
        'fft_dominant_freq': freqs[idx],
        'fft_dominant_power': power[idx],
        'fft_spectral_entropy': entropy(p_norm),
        'fft_low_freq_power': power[freqs<low_freq_cut].sum(),
        'fft_high_freq_power': power[freqs>high_freq_cut].sum(),
        'fft_power_ratio': power[freqs<low_freq_cut].sum()/(power[freqs>high_freq_cut].sum()+1e-6)
    }


def make_transaction_features(tx: pd.DataFrame) -> pd.DataFrame:
    print("Computing aggregate transaction features (agg_ prefix)...")
    tx['posted_date'] = pd.to_datetime(tx['posted_date'])
    agg = tx.groupby('masked_consumer_id').agg(
        total_amount=('amount','sum'),
        mean_amount=('amount','mean'),
        std_amount=('amount','std'),
        min_amount=('amount','min'),
        max_amount=('amount','max'),
        median_amount=('amount','median'),
        transaction_count=('amount','count'),
        credit_sum=('amount', lambda x: x[x>0].sum()),
        debit_sum=('amount', lambda x: abs(x[x<0].sum()))
    )
    agg['credit_debit_ratio'] = (agg['credit_sum']/agg['debit_sum'].replace(0,np.nan)).fillna(0)
    agg.columns = [f'agg_{c}' for c in agg.columns]
    print(f" -> {agg.shape}")
    return agg.fillna(0)


def make_weekly_features(tx: pd.DataFrame,
                         top_categories: Optional[list]=None,
                         include_cv: bool=True,
                         include_trend: bool=True,
                         include_rolling: bool=True,
                         include_weekend: bool=True) -> pd.DataFrame:
    print("Computing weekly-based category features (week_ prefix)...")
    tx['posted_date'] = pd.to_datetime(tx['posted_date'])
    tx['week'] = tx['posted_date'].dt.to_period('W').apply(lambda r: r.start_time)
    if top_categories is None:
        top_categories = tx['category'].value_counts().nlargest(35).index.tolist()
    txf = tx.loc[tx['category'].isin(top_categories)].copy()
    weekly = txf.groupby(['masked_consumer_id','category','week'])['amount'].sum().reset_index()

    parts = []
    if include_cv:
        vol = weekly.groupby(['masked_consumer_id','category'])['amount'].agg(['std','mean'])
        vol['cv'] = (vol['std']/vol['mean'].replace(0,np.nan)).fillna(0)
        vol = vol.unstack(fill_value=0)
        vol.columns = [f'week_cv_cat_{cat}' for cat in vol.columns.get_level_values(1)]
        parts.append(vol)
    if include_trend:
        trends = weekly.groupby(['masked_consumer_id','category']).apply(
            lambda df: LinearRegression().fit(
                (df['week']-df['week'].min()).dt.days.values.reshape(-1,1), df['amount']).coef_[0]
            if len(df)>=2 else 0
        ).unstack(fill_value=0)
        trends.columns = [f'week_trend_cat_{cat}' for cat in trends.columns]
        parts.append(trends)
    if include_weekend:
        txf['is_weekend'] = txf['week'].dt.weekday>=5
        weekend = txf.groupby(['masked_consumer_id','category']).apply(
            lambda df: df.loc[df['is_weekend'],'amount'].sum()/(df.loc[~df['is_weekend'],'amount'].sum()+1e-6)
        ).unstack(fill_value=0)
        weekend.columns = [f'week_weekend_ratio_cat_{cat}' for cat in weekend.columns]
        parts.append(weekend)
    if include_rolling:
        roll = weekly.copy()
        roll['rolling_mean'] = roll.groupby(['masked_consumer_id','category'])['amount'].transform(lambda x: x.rolling(3,min_periods=1).mean())
        roll['weekly_change'] = roll.groupby(['masked_consumer_id','category'])['amount'].pct_change().replace([np.inf,-np.inf],0).fillna(0)
        agg_roll = roll.groupby(['masked_consumer_id','category']).agg({'rolling_mean':'mean','weekly_change':['mean','std']})
        agg_roll.columns = [f'week_roll_{stat}_cat_{cat}' for stat,cat in agg_roll.columns]
        agg_roll = agg_roll.unstack(fill_value=0)
        parts.append(agg_roll)

    final = pd.concat(parts, axis=1).fillna(0)
    print(f" -> {final.shape}")
    return final


def compute_fft_for_tx(tx_consumer: pd.DataFrame) -> Dict[str,float]:
    feats = {}
    for cat, sub in tx_consumer.groupby('category'):
        ts = weekly_trimmed(sub.set_index('posted_date')['amount'])
        if len(ts)<6 or ts.sum()==0:
            continue
        ff = extract_fft_features(ts)
        if ff:
            feats.update({f'{k}_cat{cat}': v for k,v in ff.items()})
    return feats


def main():
    consumer_file = '/Users/jasonc/Desktop/DSC_291/cashflow/consumer_data.parquet'
    tx_file = '/Users/jasonc/Desktop/DSC_291/cashflow/transactions.parquet'
    out_file = '/Users/jasonc/Desktop/DSC_291/merged_features.parquet'

    consumer = read_parquet_auto(consumer_file).set_index('masked_consumer_id')
    tx = read_parquet_auto(tx_file)
    print("Merging evaluation_date into tx if present...")
    if 'evaluation_date' in consumer.columns:
        tx['posted_date'] = pd.to_datetime(tx['posted_date'])
        tx = tx.merge(consumer['evaluation_date'], left_on='masked_consumer_id', right_index=True)

    consumer = consumer[consumer.index.str.startswith('C01')]
    tx = tx[tx['masked_consumer_id'].isin(consumer.index)]
    print(f"Filtered: {consumer.shape[0]} consumers, {tx.shape[0]} transactions")

    # Real features
    agg_df    = make_transaction_features(tx)
    week_df   = make_weekly_features(tx)
    print("Computing real FFT features...")
    fft_rows  = []
    for cid in consumer.index:
        feats = compute_fft_for_tx(tx[tx['masked_consumer_id']==cid])
        if feats:
            feats['masked_consumer_id'] = cid
            fft_rows.append(feats)
    fft_df = pd.DataFrame(fft_rows).set_index('masked_consumer_id').fillna(0)
    print(f"FFT features: {fft_df.shape}")

    real = consumer.join(agg_df, how='inner')
    real = real.join(week_df, how='left')
    real = real.join(fft_df, how='left').fillna(0)
    real['dataset_type'] = 'real'
    print(f"Real merged: {real.shape}")

    # Simulations
    sim_list = []
    pos_ids = real[real['FPF_TARGET']==1].index
    print(f"Simulating {N_SIMULATIONS} per {len(pos_ids)} positive IDs...")
    for cid in pos_ids:
        tx_cons = tx[tx['masked_consumer_id']==cid]
        eval_date = consumer.loc[cid, 'evaluation_date'] if 'evaluation_date' in consumer.columns else pd.NaT
        for sim in range(1, N_SIMULATIONS+1):
            sim_tx = tx_cons.copy()
            noise_scale = sim_tx['amount'].std()*0.1
            sim_tx['amount'] = sim_tx['amount'] + np.random.normal(0, noise_scale, size=sim_tx.shape[0])
            agg_sim  = make_transaction_features(sim_tx).loc[cid].to_dict()
            week_sim = make_weekly_features(sim_tx).loc[cid].to_dict()
            fft_sim  = compute_fft_for_tx(sim_tx)
            sim_row  = {**{'masked_consumer_id': f"{cid}_simulation_{sim}",
                            'evaluation_date': eval_date,
                            'FPF_TARGET':1,
                            'dataset_type':'simulation'},
                        **agg_sim, **week_sim, **fft_sim}
            sim_list.append(sim_row)
    sim_df = pd.DataFrame(sim_list).set_index('masked_consumer_id').fillna(0)
    print(f"Simulated rows: {sim_df.shape}")

    # Final concat
    real   = real.loc[:, ~real.columns.duplicated()]
    sim_df = sim_df.loc[:, ~sim_df.columns.duplicated()]
    final  = pd.concat([real, sim_df], axis=0).fillna(0)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    final.to_parquet(out_file)
    print(f"Saved final dataset: {final.shape} -> {out_file}")

if __name__ == '__main__':
    main()
