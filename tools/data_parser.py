import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import pywt

col_types = ['int64', 'int64', 'int64', 'int64', 'int64', 'float64', 'int64', 'float64', 'int64', 'float64',
             'int64', 'float64', 'int64', 'float64', 'int64', 'float64', 'float64', 'int64', 'float64', 'float64',
             'float64', 'float64', 'int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'bool']

def remove_duplicated_values(df: pd.DataFrame):
    init_row_cnt = len(df)
    df.drop_duplicates(subset='TIMESTAMP', inplace=True)
    print('DUPLICATE DATES removed: ', init_row_cnt-len(df))

def remove_pit_suffix(name: str) -> str:
    """
    Remove suffix '_pit<number>' from header
    """
    re_match = re.search(r'_pit\d+$', name)
    if re_match:
        name = name[:re_match.start()]
    return name

def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix data types from given DataFrame
    """
    for col_type, col_name in zip(col_types, list(df.columns.array)[1:]):
        if col_type == 'int64':
            df[col_name] = df[col_name].astype('float64').astype('int64')
        elif col_type == 'float64':
            df[col_name] = df[col_name].astype('float64')
        else:
            df[col_name] = df[col_name].astype('bool')
    return df

def prune_unnecessary_features(df: pd.DataFrame):
    unnecessary_features = ['Temp_T21_Avg(1)','Temp_T21_Avg(2)', 'Temp_T21_Avg(3)', 'Temp_T21_Avg(4)', 'Temp_T21_Avg(5)',
                            'CCVWC_Avg(1)', 'CCVWC_Avg(2)', 'CCVWC_Avg(3)', 'CCVWC_Avg(4)', 'CCVWC_Avg(5)', 'shf_plate_Avg',
                            'shf_multiplier', 'shf_htr_resstnc', 'shfp_wrnng_flg', 'btt_wrnng_flg', 'PTemp_C_Avg', 'RECORD', 'BattV_Min']
    df.drop(unnecessary_features, axis=1, inplace=True)

def get_time_windows(hour_prediods: list[int]):
    return [int((period_h*60)/5) for period_h in hour_prediods]

def timestamp_gap(df: pd.DataFrame, td: timedelta) -> tuple: 
    df["TIMESTAMP_DIFF"] = df.loc[:, "TIMESTAMP"].diff()
    
    breaks = list(df.index[(df["TIMESTAMP_DIFF"] != td) & (np.isnan(df["TIMESTAMP_DIFF"]) == False)])
    gaps = [int(df.loc[index, "TIMESTAMP_DIFF"].total_seconds()/60) for index in breaks]
    
    return (breaks, gaps)

def calculate_backward_sigma(rolling_window: pd.DataFrame, breaks: list, window_size: int, starting_index: int):
    if rolling_window.index[-1] < starting_index + window_size-1:
        return([np.nan, np.nan, np.nan, np.nan, np.nan])
    for j in breaks:
        break_zero_index = rolling_window.index[-1] - j
        if (break_zero_index >=0) and (break_zero_index < window_size-1):
            return([np.nan, np.nan, np.nan, np.nan, np.nan])
    return rolling_window[["Redox_Avg(1)", "Redox_Avg(2)", "Redox_Avg(3)", "Redox_Avg(4)", "Redox_Avg(5)"]].apply(func=np.std, axis=0)

def sigma_feature_engineering(df: pd.DataFrame, window_sizes, td):
    breaks = timestamp_gap(df, td)[0]

    for window_size in window_sizes:
        widow_h = int((window_size*5)/60)
        for sensor in range(1,6):
            df[f'Redox_Avg({sensor})_sigma_b_{widow_h}'] = np.nan
            df[f'Redox_Avg({sensor})_sigma_f_{widow_h}'] = np.nan
        
        backward_sigma = np.array([calculate_backward_sigma(rolling_window, breaks = breaks, window_size = window_size, starting_index = df.index[0]) for rolling_window in df.rolling(window_size)])

        df[[f'Redox_Avg(1)_sigma_b_{widow_h}', f'Redox_Avg(2)_sigma_b_{widow_h}', f'Redox_Avg(3)_sigma_b_{widow_h}', f'Redox_Avg(4)_sigma_b_{widow_h}', f'Redox_Avg(5)_sigma_b_{widow_h}']] = backward_sigma
        
        forward_sigma = df.loc[:, [f'Redox_Avg(1)_sigma_b_{widow_h}', f'Redox_Avg(2)_sigma_b_{widow_h}', f'Redox_Avg(3)_sigma_b_{widow_h}', f'Redox_Avg(4)_sigma_b_{widow_h}', f'Redox_Avg(5)_sigma_b_{widow_h}']].shift(-(window_size-1))
        df[[f'Redox_Avg(1)_sigma_f_{widow_h}', f'Redox_Avg(2)_sigma_f_{widow_h}', f'Redox_Avg(3)_sigma_f_{widow_h}', f'Redox_Avg(4)_sigma_f_{widow_h}', f'Redox_Avg(5)_sigma_f_{widow_h}']] = forward_sigma.to_numpy()

    sigma_col_names_list = [col_name for col_name in df.columns.array if 'sigma' in col_name]
    removed_rows = df.index[np.any(df.loc[:, sigma_col_names_list].isna(), axis=1)]

    df.drop(removed_rows, axis=0, inplace=True)

    print(f"{len(removed_rows)} rows had to be removed due to \" hitting \" time gaps or margin areas when calculating standard deviation")
    
    return df

def calculate_wavelet_coefficients(df: pd.DataFrame, period_lower_bound: float, period_upper_bound: float) -> None:
    # period of the wave T is calculated in days, frequency = 1/T 
    dt = (datetime(2023,1,1,0,5,0) - datetime(2023,1,1,0,0,0)).seconds / (60 * 60 * 24)

    for sensor in np.arange(5):
        redox_series = "Redox_Avg(" + str(sensor + 1) + ")"     
        scales = np.geomspace(1, 2400, 30)
        signal = df[redox_series]
        [coefficients, frequencies] = pywt.cwt(signal, scales, "cmor1.5-0.5", dt)
        power = abs(coefficients)
        periods = 1 / frequencies
        coef_idx = np.where((periods >= period_lower_bound) & (periods <= period_upper_bound), True, False)
        power = power[np.arange(len(frequencies))[coef_idx], :].T
        wavelet_cols = ["Wave_period_" + str(round(period,1)) + "(" + str(sensor + 1) + ")" for period in periods[np.arange(len(frequencies))[coef_idx]]]
        wavelet_df = pd.DataFrame(power, columns = wavelet_cols, index = df.index)
        df = pd.concat([df, wavelet_df], axis = 1)
    
    return df

def parse_raw_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, parse_dates=['TIMESTAMP'])

    # Rename columns for prediction
    df.rename(mapper=remove_pit_suffix, axis='columns', inplace=True)
    
    # Remove features not used in prediction and unkown types
    prune_unnecessary_features(df)
    
    # Fix feature types
    df = fix_types(df)

    # Remove duplicated dates
    remove_duplicated_values(df)

    # Add sigma features
    window_sizes = get_time_windows([12, 24])
    td = timedelta(days = 0, hours = 0, minutes = 5, seconds = 0, milliseconds= 0, weeks = 0)
    df = sigma_feature_engineering(df, window_sizes=window_sizes, td=td)

    # Add wavelet features
    df = calculate_wavelet_coefficients(df, 1/2, 5)

    return df