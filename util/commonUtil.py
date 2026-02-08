import pandas as pd
import numpy as np
from datetime import datetime


def get_data_source(file_path):
    """读取CSV数据"""
    return pd.read_csv(file_path)


def format_time(dt):
    """格式化时间为 2025-08-04 09:00:00"""
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def sort_by_time(df, time_col):
    """按时间列升序排序"""
    df[time_col] = pd.to_datetime(df[time_col])
    return df.sort_values(time_col).reset_index(drop=True)


def deduplicate(df, subset=None, keep='first'):
    """去重"""
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100