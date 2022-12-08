import pandas as pd
import holidays
import numpy as np

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))




def preprocess(df, datecol='datetime'):
    yy = df.set_index(datecol)
    yy['days_in_month'] = yy.index.days_in_month
    yy['day'] = yy.index.day
    yy['week_day_of_year'] = pd.Int64Index(yy.index.isocalendar().week)
    yy['day_of_year'] = yy.index.dayofyear
    yy['holiday'] = yy.index.map(lambda x: int(x in holidays.IN()))
    yy["week"] = pd.Int64Index(yy.index.isocalendar().week)
    yy["month"] = yy.index.month
    yy["quarter"] = yy.index.quarter
    yy["year"] = yy.index.year
    yy["dayofyear"] = yy.index.dayofyear
    yy['day_of_week'] = yy.index.day_of_week.astype(int)
    yy["is_month_start"] = yy.index.is_month_start.astype(int)
    yy["is_month_end"] = yy.index.is_month_end.astype(int)
    yy["is_quarter_start"] = yy.index.is_quarter_start.astype(int)
    yy["is_quarter_end"] = yy.index.is_quarter_end.astype(int)
    yy["is_year_start"] = yy.index.is_year_start.astype(int)
    yy["is_year_end"] = yy.index.is_year_end.astype(int)
    yy["is_leap_year"] = yy.index.is_leap_year.astype(int)
    yy['is_weekend'] = np.where(yy['day_of_week'].isin([5, 6]), 1, 0)
    yy["sin_week"] = sin_transformer(7).fit_transform(yy['week'])
    yy["sin_month"] = sin_transformer(12).fit_transform(yy['month'])
    yy["sin_quarter"] = sin_transformer(4).fit_transform(yy['quarter'])
    yy["sin_dayofyear"] = sin_transformer(365).fit_transform(yy['dayofyear'])

    yy["cos_week"] = cos_transformer(7).fit_transform(yy['week'])
    yy["cos_month"] = cos_transformer(12).fit_transform(yy['month'])
    yy["cos_quarter"] = cos_transformer(4).fit_transform(yy['quarter'])
    yy["cos_dayofyear"] = cos_transformer(365).fit_transform(yy['dayofyear'])
    return yy


def loaddf(path='train_IxoE5JN.csv'):
    df = pd.read_csv(path)
    df = df.bfill().ffill()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.drop(['row_id'], axis=1)
    return df


def splitdf(df, time_horizon):
    data = df[['datetime', 'energy']]
    num_samples = data.shape[0]
    # time_horizon = 100
    split_idx = num_samples - time_horizon
    # train_df is a dataframe with two columns: timestamp and label
    train_df = data[:split_idx]
    # X_test is a dataframe with dates for prediction
    X_test = data[split_idx:]['datetime'].to_frame()
    y_test = data[split_idx:]['energy']
    return train_df, X_test, y_test


if __name__ == "__main__":
    pass
