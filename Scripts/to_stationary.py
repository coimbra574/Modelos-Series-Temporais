
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.preprocessing import StandardScaler




"""# 2 - Serie apenas normalizada"""

def normalize(df):
    scaler = StandardScaler()
    scaled_carga = scaler.fit_transform(df['carga'].values.reshape(-1,1))
    dates = df['data'].values
    df_norm = pd.DataFrame({'carga' : scaled_carga.reshape(-1,), 'data' : dates})
    return df_norm


"""# 3 - Serie original sem tendencia por ajuste de curva"""

def trend_fit_curve(df):
    # Como dados estao em Datetime, o eixo x sera a diferenca entre os tempos
    x = np.array([x/(24*3600*(10**9)) for x in np.diff(np.array(df['data']))]).cumsum()
    x = np.concatenate((np.array([0]), x), axis=0).astype('int')

    # Fit curve
    poly = np.polyfit(x, df['carga'].values, deg=5)

    # Predict
    pred_curve = np.polyval(poly, x)
    df_pred_curve = pd.DataFrame(pred_curve, columns=['carga']).set_index(df.index)

    df_trend_fit = pd.DataFrame()
    df_trend_fit['data'] = df['data'].copy()
    df_trend_fit['carga'] = df['carga'] - df_pred_curve['carga']

    return df_trend_fit, np.poly1d(poly)



"""# 4 - Serie original sem tendencia por diferenciacao"""

def trend_diff(df):
    df_trend_diff = pd.DataFrame()
    df_trend_diff['data'] = df['data'].copy()
    df_trend_diff['carga'] = df['carga'].diff().copy()
    df_trend_diff.loc[df.index[0], 'carga'] = df_trend_diff.loc[df.index[0]+1, 'carga']
    return df_trend_diff



"""# 5 - Serie original sem sazonalidade anual e semanal por padronizacao

Padronizado por cada dia do ano -> Ex: 01.01
"""

def season_padronization(df, period):
    month = df['data'].dt.month.values
    day = df['data'].dt.day.values
    weekday = df['data'].dt.weekday.values
    df_temp = pd.DataFrame({'data': df.data, 'carga': df.carga, 'month': month, 'day': day, 'weekday': weekday})

    if period == 'year':
        index = ['day','month']
    elif period == 'week':
        index = 'weekday'
    else:
        print("Not a valid period. Try year or week")
        return

    # Define custom std function, since np.std does not work on pivot_table if we have only 1 value
    def custom_std(x):
        if len(x) == 1:
            return 1  # Return 0 or any desired value for single values
        else:
            return np.std(x)

    # Get mean carga values for each day of the week
    period_mean = pd.pivot_table(df_temp, index=index, values=['carga'], aggfunc=np.mean)
    period_std = pd.pivot_table(df_temp, index=index, values=['carga'], aggfunc=custom_std)

    ## Normalize values by each day of the week
    def normalize(row, period_mean, period_std, period):
        if period == 'year':
            norm_row = (row.carga - period_mean.loc[row.day, row.month]) / period_std.loc[row.day, row.month]
        else:
            norm_row = (row.carga - period_mean.iloc[row.weekday]) / period_std.iloc[row.weekday]
        return norm_row
        
    df_desazon = pd.DataFrame()
    df_desazon['data'] = df_temp['data'].copy()
    df_desazon['carga'] = df_temp.apply(lambda row: normalize(row, period_mean, period_std, period), axis=1)
    return df_desazon, period_mean, period_std



"""# 6 - Serie original sem sazonalidade anual e semanal por medias moveis

Medias moveis com base na media geral dos dias de cada mes

"""

def season_MA(df, step):
    if step == 'month':
        values = df['data'].dt.month
    elif step == 'weekday':
        values = df['data'].dt.weekday
    else:
        return
    df_temp = pd.DataFrame({'data': df.data, 'carga': df.carga, step: values})

    mean_values = pd.pivot_table(df_temp, index=step, values=['carga'], aggfunc=np.mean)
    mean_values['carga'] = mean_values['carga'] - mean_values['carga'].mean()

    def MA(row, mean_values):
        if step == 'month':
            return row.carga - mean_values.iloc[row.month-1]
        else:
            return row.carga - mean_values.iloc[row.weekday] 
  
    df_deseason = pd.DataFrame()
    df_deseason['data'] = df_temp['data'].copy()
    df_deseason['carga'] = df_temp.apply(lambda row: MA(row, mean_values), axis=1)
    return df_deseason, mean_values




"""# 7 - Serie original sem sazonalidade anual e semanal por diferenca sazonal

x_t = x_t - x_{t+L}

L = step sazonal 
"""

def season_diff(df, shift_step):
    df_shifted = df['carga'].shift(shift_step).fillna(0)
    df_desazon = pd.DataFrame()
    df_desazon['data'] = df['data'].copy()
    df_desazon['carga'] = df['carga'] - df_shifted
    return df_desazon
