
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from datetime import timedelta
from xgboost import XGBClassifier
import os
from scipy.stats import entropy
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectFromModel, mutual_info_classif, f_classif, SelectKBest
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb


def model_1(transactions_test, consumer_test):
    #generate features
    model = xgb.Booster({'nthread': 4})
    model.load_model('Final_workflow/models/model01.json')
    X = make_transaction_features_with_weekly(transactions_test, list(transactions_test['masked_consumer_id'].unique()))
    X_val = consumer_test.merge(X, on='masked_consumer_id', how='left')
    X_dropped = X_val.drop(columns=['masked_consumer_id', 'evaluation_date']).fillna(0)
    model_features = model.feature_names 
    _, X_dropped = pd.read_csv("Final_workflow/models/train_features_c01.csv").align(X_dropped, join='outer', axis=1, fill_value=0) 
    X_dropped = X_dropped[model_features] 
    dmatrix = xgb.DMatrix(X_dropped)
    pred = model.predict(dmatrix) 
    return pd.Series(pred, index=X_val['masked_consumer_id'], name='prediction')


def make_transaction_features(transactions):
    tx = transactions.copy()

    # Create debit/credit indicators
    tx['is_credit'] = tx['amount'] > 0
    tx['is_debit'] = tx['amount'] < 0

    # Basic aggregate features
    agg = tx.groupby('masked_consumer_id').agg(
        total_amount=('amount', 'sum'),
        mean_amount=('amount', 'mean'),
        std_amount=('amount', 'std'),
        min_amount=('amount', 'min'),
        max_amount=('amount', 'max'),
        median_amount=('amount', 'median'),
        transaction_count=('amount', 'count'),
        credit_sum=('is_credit', lambda x: tx.loc[x.index, 'amount'][x].sum()),
        debit_sum=('is_debit', lambda x: abs(tx.loc[x.index, 'amount'][x].sum()))
    )
    agg['credit_debit_ratio'] = agg['credit_sum'] / agg['debit_sum'].replace(0, np.nan)
    agg['credit_debit_ratio'] = agg['credit_debit_ratio'].fillna(0)

    # Recent 30-day features
    tx['days_before_eval'] = (tx['evaluation_date'] - tx['posted_date']).dt.days
    recent_df = tx[tx['days_before_eval'] <= 30].groupby('masked_consumer_id').agg(
        recent30_sum=('amount', 'sum'),
        recent30_count=('amount', 'count')
    )

    # Monthly volatility
    tx['month'] = tx['posted_date'].dt.to_period('M')
    monthly_cashflow = tx.groupby(['masked_consumer_id', 'month'])['amount'].sum().reset_index()
    monthly_stats = monthly_cashflow.groupby('masked_consumer_id')['amount'].agg(['mean', 'std']).rename(
        columns={'mean': 'monthly_mean', 'std': 'monthly_std'})
    monthly_stats['monthly_cv'] = monthly_stats['monthly_std'] / monthly_stats['monthly_mean'].replace(0, np.nan) # coefficient of variation
    monthly_stats = monthly_stats.fillna(0)

    # Weekend spending
    tx['weekday'] = tx['posted_date'].dt.weekday
    tx['is_weekend'] = tx['weekday'] >= 5

    # Calculate weekend and weekday spending separately
    weekend_spend = tx[tx['amount'] > 0].groupby('masked_consumer_id').agg(
        weekday_spending=('amount', lambda x: x[~tx.loc[x.index, 'is_weekend']].sum()),
        weekend_spending=('amount', lambda x: x[tx.loc[x.index, 'is_weekend']].sum())
    )

    weekend_spend['weekend_ratio'] = weekend_spend['weekend_spending'] / (weekend_spend['weekday_spending'] + 1e-6)

    # Transaction frequency
    txn_freq = tx.groupby('masked_consumer_id').agg(
        unique_txn_days=('posted_date', lambda x: x.nunique()),
        txn_days_span=('posted_date', lambda x: (x.max() - x.min()).days + 1),
    )
    txn_freq['txn_per_day'] = txn_freq['unique_txn_days'] / txn_freq['txn_days_span'].replace(0, 1)

    # Percentiles
    percentiles = tx.groupby('masked_consumer_id')['amount'].quantile([0.25, 0.75, 0.9]).unstack()
    percentiles.columns = ['amount_25pct', 'amount_75pct', 'amount_90pct']

    # Large transactions
    tx['is_large'] = tx['amount'].abs() > 1000
    large_txns = tx.groupby('masked_consumer_id')['is_large'].sum().rename('large_txn_count')

    # Monthly trend
    monthly_spend = tx.groupby(['masked_consumer_id', 'month'])['amount'].sum().reset_index()
    monthly_spend['month'] = monthly_spend['month'].dt.to_timestamp()

    def compute_monthly_trend(df):
        if df.shape[0] < 2:
            return 0
        df = df.sort_values('month')
        x = (df['month'] - df['month'].min()).dt.days.values.reshape(-1, 1)
        y = df['amount'].values
        model = LinearRegression().fit(x, y)
        return model.coef_[0]

    monthly_trend = monthly_spend.groupby('masked_consumer_id').apply(compute_monthly_trend).rename('monthly_spend_trend')

    # Category encoding
    cat_encode = pd.get_dummies(tx, columns=['category'], drop_first=False)
    cat_cols = [col for col in cat_encode.columns if col.startswith('category_')]
    cat_encode_mean = cat_encode.groupby('masked_consumer_id')[cat_cols].sum()

    # Category-Specific Spending
    cat_spending = tx.pivot_table(index='masked_consumer_id', columns='category', values='amount', aggfunc='sum', fill_value=0)
    cat_spending.columns = [f'spend_cat_{int(c)}' for c in cat_spending.columns]

    # Credit-Debit Gap per Category
    def credit_debit_gap(df):
        pos = df[df['amount'] > 0].groupby('category')['amount'].sum()
        neg = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        return (pos - neg).fillna(0)

    gap_df = tx.groupby('masked_consumer_id').apply(credit_debit_gap).unstack().add_prefix('gap_cat_').fillna(0)

    # Binary Category Usage
    usage = pd.crosstab(tx['masked_consumer_id'], tx['category']).astype(bool).astype(int)
    usage.columns = [f'used_cat_{int(c)}' for c in usage.columns]

    # Timing Features per Category
    cat_days = tx.groupby(['masked_consumer_id', 'category'])['posted_date'].agg(['min', 'max', 'count'])
    cat_days['duration'] = (cat_days['max'] - cat_days['min']).dt.days
    cat_days['txn_gap'] = cat_days['duration'] / cat_days['count'].replace(0, 1)
    duration_feat = cat_days['txn_gap'].unstack().fillna(0).add_prefix('gap_days_cat_')

    # Merge all features
    features_df = agg \
        .join(recent_df, how='left') \
        .join(monthly_stats, how='left') \
        .join(txn_freq, how='left') \
        .join(weekend_spend, how='left') \
        .join(large_txns, how='left') \
        .join(percentiles, how='left') \
        .join(monthly_trend, how='left') \
        .join(cat_encode_mean, how='left') \
        .join(gap_df, how='left') \
        .join(usage, how='left') \
        .join(duration_feat, how='left') \
        .fillna(0) \
        .reset_index()

    return features_df

def make_weekly_features(
    transactions,
    top_categories=None,
    include_cv=True,
    include_trend=True,
    include_rolling=True,
    include_weekend=True
):

    tx = transactions.copy()
    tx['week'] = tx['posted_date'].dt.to_period('W').apply(lambda r: r.start_time)

    if top_categories is None:
        top_categories = tx['category'].value_counts().nlargest(35).index.tolist()
    tx = tx[tx['category'].isin(top_categories)]

    # Base weekly aggregation
    weekly_stats = tx.groupby(['masked_consumer_id', 'category', 'week'])['amount'].agg(['sum', 'mean', 'std', 'count']).reset_index()
    weekly_stats = weekly_stats.sort_values(['masked_consumer_id', 'category', 'week'])

    if include_rolling:
        weekly_stats['rolling_mean'] = weekly_stats.groupby(['masked_consumer_id', 'category'])['sum'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        weekly_stats['rolling_std'] = weekly_stats.groupby(['masked_consumer_id', 'category'])['sum'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))

        weekly_stats['weekly_change'] = (
            weekly_stats.groupby(['masked_consumer_id', 'category'])['sum']
            .pct_change()
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )

    # Compute CV
    if include_cv:
        cat_volatility = weekly_stats.groupby(['masked_consumer_id', 'category'])['sum'].agg(['std', 'mean'])
        cat_volatility['cv'] = cat_volatility['std'] / cat_volatility['mean'].replace(0, np.nan)
        cat_volatility = cat_volatility.unstack().fillna(0)
        cat_volatility.columns = [f'weekly_cat_{stat}_{int(cat)}' for stat, cat in cat_volatility.columns]
    else:
        cat_volatility = None

    # Compute linear trend
    if include_trend:
        def compute_weekly_trend(df):
            if df.shape[0] < 2:
                return 0
            df = df.sort_values('week')
            x = (df['week'] - df['week'].min()).dt.days.values.reshape(-1, 1)
            y = df['sum'].values
            model = LinearRegression().fit(x, y)
            return model.coef_[0]

        cat_trends = weekly_stats.groupby(['masked_consumer_id', 'category']).apply(compute_weekly_trend).unstack().fillna(0)
        cat_trends.columns = [f'weekly_trend_cat_{int(c)}' for c in cat_trends.columns]
    else:
        cat_trends = None

    # Weekend vs weekday spending
    if include_weekend:
        tx['weekday'] = tx['posted_date'].dt.weekday
        tx['is_weekend'] = tx['weekday'] >= 5
        cat_weekend = tx.groupby(['masked_consumer_id', 'category']).apply(
            lambda df: df[df['is_weekend']]['amount'].sum() / (df[~df['is_weekend']]['amount'].sum() + 1e-6)
        ).unstack().fillna(0)
        cat_weekend.columns = [f'weekend_ratio_cat_{int(c)}' for c in cat_weekend.columns]
    else:
        cat_weekend = None

    # Initialize aggregation dict
    agg_dict = {}
    if include_rolling:
        agg_dict.update({
            'rolling_mean': 'mean',
            'rolling_std': 'mean',
            'weekly_change': ['mean', 'std']
        })
    agg_dict['count'] = 'mean'

    # Perform aggregation only if dict is not empty
    if agg_dict:
        feats = weekly_stats.groupby(['masked_consumer_id', 'category']).agg(agg_dict)
        feats.columns = [f'{col}_{stat}' for col, stat in feats.columns]
        feats = feats.unstack(level=1).fillna(0)
        feats.columns = [f'{col}_cat_{int(cat)}' for col, cat in feats.columns]
    else:
        feats = pd.DataFrame(index=weekly_stats['masked_consumer_id'].unique())


    # Combine all parts
    final_feats = feats.reset_index()

    if include_cv and cat_volatility is not None:
        final_feats = final_feats.merge(cat_volatility.reset_index(), on='masked_consumer_id', how='left')
    if include_trend and cat_trends is not None:
        final_feats = final_feats.merge(cat_trends.reset_index(), on='masked_consumer_id', how='left')
    if include_weekend and cat_weekend is not None:
        final_feats = final_feats.merge(cat_weekend.reset_index(), on='masked_consumer_id', how='left')

    final_feats = final_feats.fillna(0)

 
    return final_feats

def make_transaction_features_with_weekly(transactions, consumer_ids):
    df = transactions[transactions['masked_consumer_id'].isin(consumer_ids)]
    base_feats = make_transaction_features(df)
    weekly_feats = make_weekly_features(df)
    final = base_feats \
    .merge(weekly_feats, on='masked_consumer_id', how='left') \
    .fillna(0)
    cats = transactions['category'].unique()
    for cat in cats:
        cat_df = df[df['category'] == cat]
        if not cat_df.empty:
            cat_feats = make_transaction_features(cat_df, category=True)
            final = final.merge(cat_feats, on='masked_consumer_id', how='left', suffixes=('', f'_cat_{cat}'))
    return final

def model_pred(model, X_val, threshold=0.5):
    y_val_proba = model.predict_proba(X_val)[:, 1]
    predictions = (y_val_proba > threshold).astype(int)
    return predictions
