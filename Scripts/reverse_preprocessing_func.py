import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def compare_plots(df_original, df):
    plt.figure(figsize=(10,4))
    plt.plot(df_original['data'], df_original['carga'])
    plt.plot(df['data'], df['carga'])
    plt.legend(['Serie original', 'Serie a comparar'])
    plt.grid()
    plt.show()



"""# 2 . df_sudeste_norm"""

def reverse_norm(df_original, df_norm):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_original['carga'].values.reshape(-1,1))
    carga = scaler.inverse_transform(df_norm['carga'].values.reshape(-1,1))
    df_denorm = pd.DataFrame({'carga' : list(carga.reshape(-1,))}, index=df_norm.index)
    df_denorm['data'] = df_norm['data'].copy()
    return df_denorm



"""# 3 - Série original sem tendência por ajuste de curva"""

def reverse_trend_fit(df, poly_coefs):

    # Como dados estao em Datetime, o eixo x será a diferença entre os tempos
    # Será considerado aqui o intervalo de tempo de df, a série que queremos remover a tendência
    # Se tivermos fazendo uma previsão, a data não deve começar de 0
    x = np.array([x/(24*3600*(10**9)) for x in np.diff(np.array(df['data']))]).cumsum() 
    x = np.concatenate((np.array([0]), x), axis=0).astype('int')

    df_new = df.copy()
    eq = np.poly1d(poly_coefs)
    trend = np.polyval(eq, x)
    df_new['carga'] += trend
    return df_new



"""# 4 - Série original sem tendência por diferenciação"""

def reverse_trend_diff(df_original, df_pred):
    initial_date = df_pred['data'].iloc[0]
    final_date = df_pred['data'].iloc[-1]
    
    mask = (df_original['data'] >= initial_date) & (df_original['data'] <= final_date)
    df_masked = df_original[mask]
    df_pred = df_pred.set_index(df_masked.index)
    carga_denorm = df_masked['carga'].shift(1).fillna(0) + df_pred['carga']
    
    df_denorm = pd.DataFrame({'data': df_pred.data, 'carga': carga_denorm})
    df_denorm.loc[df_denorm.index[0], 'carga'] = df_masked.loc[df_masked.index[0], 'carga']
    return df_denorm


"""# 5 - Série original sem sazonalidade anual e semanal por padronização"""

def add_sazon_pad(df, mean_pad, std_pad, period):    
    def add_sazon_week(row, week_mean, week_std):
        return row.carga * week_std.iloc[row.weekday] + week_mean.iloc[row.weekday]

    def add_sazon_anual(row, day_mean, day_std):
        return row.carga * day_std.loc[row.day, row.month] + day_mean.loc[row.day, row.month]

    # Add sazonalidade semanal 
    weekday = df['data'].dt.weekday
    day = df['data'].dt.day
    month = df['data'].dt.month
    df_temp = pd.DataFrame({'data': df.data, 'carga': df.carga, 'weekday': weekday, 'day': day, 'month': month})

    if period == 'year':
        add_sazon = add_sazon_anual
    elif period == 'week':
        add_sazon = add_sazon_week
    else:
        return

    df_desazon  = pd.DataFrame({'data': df_temp.data})
    df_desazon['carga'] = df_temp.apply(lambda row: add_sazon(row, mean_pad, std_pad), axis=1)
    return df_desazon



"""# 6 - Série original sem sazonalidade anual e semanal por médias móveis"""

def add_sazon_MA(df, mean_values, step):
    if step == 'month':
        df[step] = df['data'].dt.month
    elif step == 'weekday':
        df[step] = df['data'].dt.weekday
    else:
        return

    def MA(row, mean_values):
        if step == 'month':
            return row.carga + mean_values.iloc[row.month-1]
        else:
            return row.carga + mean_values.iloc[row.weekday] 
  
    df_back = pd.DataFrame()
    df_back['data'] = df['data'].copy()
    df_back['carga'] = df.apply(lambda row: MA(row, mean_values), axis=1)
    return df_back



"""# 7 - Série original sem sazonalidade anual e semanal por diferença sazonal"""

# A série sem sazonalidade semanal está como yt0 + yt1 + ... yt6 + (yt7-yt0) + (yt8-yt1) + ... + ytL - ytL-7.
# Então uma forma de desazonalizar é criar um section inc que guarda os valores das seções (no primeiro caso, eg [yt0, yt1, ... yt6]) 
# e faz (yt7-yt0) + (yt8-yt1) + .. - yt0 - yt1 - ... . Depois section_inc guarda o resultado e usa para subtrair 
# da próxima seção e assim por diante

def add_sazon_diff(df_original, df_compare, step):
    df_compare.loc[0, 'carga'] = float(df_original[df_original['data'] == df_compare['data'][0]]['carga'])
    section_inc = np.array(df_compare['carga'][:step])
    zeros = np.zeros((len(df_compare)-len(section_inc),))
    section_inc = pd.Series(np.concatenate((section_inc,zeros)))

    df_new = pd.DataFrame(df_compare).copy()

    while (section_inc.iloc[-1] == 0):
        section_inc = section_inc.shift(step).fillna(0)
        df_new['carga'] += section_inc
        mask = ~(section_inc == 0)
        section_inc = (df_new['carga'] * mask)
    return df_new

