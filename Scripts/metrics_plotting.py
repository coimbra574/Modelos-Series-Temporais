"""
Functions for plotting and calculating metrics
"""

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd


def pred_to_dataframe(y_pred, y_date):
    df_pred = pd.DataFrame(y_pred, columns=['carga'])
    df_pred['data'] = y_date
    return df_pred


def compare_predictions(df, df_pred):
    plt.figure(figsize=(18,4))
    initial_date = df_pred['data'].iloc[0]
    final_date = df_pred['data'].iloc[-1]
    mask_date = (df['data'] >= initial_date) & (df['data'] <= final_date)
    plt.plot(df[mask_date]['data'], df[mask_date]['carga'])
    plt.plot(df_pred['data'], df_pred['carga'])
    plt.legend(['Original', 'Predição'])


def compare_predictions_zoom(y_pred_denorm, y_test_denorm, zoom='all'):
    '''
    Zoom can be a Range type, or 'all' for the whole test size

    '''

    if zoom == 'all':
        zoom = range(len(y_pred_denorm))

    plt.figure(figsize=(18,4))
    y_pred_denorm_zoom = y_pred_denorm.iloc[zoom]
    y_test_denorm_zoom = y_test_denorm.iloc[zoom]
    plt.plot(y_pred_denorm_zoom['data'], y_pred_denorm_zoom['carga'])
    plt.plot(y_test_denorm_zoom['data'], y_test_denorm_zoom['carga'])
    plt.legend(['Predição', 'Teste'])


def calc_metrics(df, df_pred):
    initial_date = df_pred['data'].iloc[0]
    final_date = df_pred['data'].iloc[-1]
    mask_date = (df['data'] >= initial_date) & (df['data'] <= final_date)
    y = df[mask_date]['carga'].values
    y_pred = df_pred['carga'].values

    return {'mse': mean_squared_error(y, y_pred), 'mape': mean_absolute_percentage_error(y, y_pred)}


def calc_mse(df, df_pred):
    initial_date = df_pred['data'].iloc[0]
    final_date = df_pred['data'].iloc[-1]
    mask_date = (df['data'] >= initial_date) & (df['data'] <= final_date)
    y = df[mask_date]['carga'].values
    y_pred = df_pred['carga'].values
    return mean_squared_error(y, y_pred)