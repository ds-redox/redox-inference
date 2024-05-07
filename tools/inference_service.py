import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def custom_scaler(df: pd.DataFrame) -> pd.DataFrame:
    """MinMax scaling for features"""
    df[list(df.columns.array)] = MinMaxScaler().fit_transform(df.loc[:, list(df.columns.array)])
    return df

def get_sensor_features(df: pd.DataFrame, redox_sensor: int) -> pd.DataFrame:
    return df.loc[:, [f'Redox_Avg({redox_sensor})', f'Temp_T12_Avg({redox_sensor})', f'EC_Avg({redox_sensor})', f'Matric_potential_Avg({redox_sensor})',
                      'Water_level_Avg', 'Temp_ottpls_Avg', 'BatterymV_Min', f'WC{redox_sensor}', f'Redox_Avg({redox_sensor})_sigma_b_24',
                      f'Redox_Avg({redox_sensor})_sigma_f_24', f'Redox_Avg({redox_sensor})_sigma_b_12', f'Redox_Avg({redox_sensor})_sigma_f_12',
                      f'Wave_period_0.5({redox_sensor})', f'Wave_period_0.7({redox_sensor})', f'Wave_period_0.9({redox_sensor})',
                      f'Wave_period_1.1({redox_sensor})', f'Wave_period_1.5({redox_sensor})', f'Wave_period_1.9({redox_sensor})',
                      f'Wave_period_2.5({redox_sensor})', f'Wave_period_3.3({redox_sensor})', f'Wave_period_4.4({redox_sensor})']]

def get_feautures(df: pd.DataFrame) -> pd.DataFrame:
    removable_columns = ['TIMESTAMP', 'TIMESTAMP_DIFF', 'Redox_error_flag']
    return df.loc[:, ~df.columns.isin(removable_columns)]

def get_sensor_data(df: pd.DataFrame, scale_data: bool) -> pd.DataFrame:
    sensor_data = dict()
    for sensor in range(1,6):
        data = get_sensor_features(df, sensor)
        if scale_data:
            data = custom_scaler(data)
        sensor_data[f'sensor_{sensor}'] = data
    return sensor_data

def get_data(df: pd.DataFrame, scale_data: bool) -> pd.DataFrame:
    data = get_feautures(df)
    if scale_data:
        data = custom_scaler(data)
    return data

def get_predictions(model, df: pd.DataFrame) -> np.ndarray:
    model_features = model.feature_names_in_
    return model.predict(df.loc[:, model_features])

def get_combined_errors(sensor_predictions: dict) -> np.ndarray:
    redox_errors = sensor_predictions['sensor_1']
    for sensor in range(2,6):
        redox_errors = np.logical_or(redox_errors, sensor_predictions[f'sensor_{sensor}'])
    return redox_errors

def sensor_model_results(models: dict, df: pd.DataFrame, scale_data: bool):
    sensor_data = get_sensor_data(df, scale_data)
    sensor_predictions = dict()
    for sensor in range(1,6):
        sensor_predictions[f'sensor_{sensor}'] = get_predictions(models[f'sensor_{sensor}'], sensor_data[f'sensor_{sensor}'])
    redox_errors = get_combined_errors(sensor_predictions)
    
    df['Redox_error_flag'] = redox_errors

def single_model_results(model, df: pd.DataFrame, scale_data: bool):
    data = get_data(df, scale_data)
    predictions = get_predictions(model, data)
    df['Redox_error_flag'] = predictions

def get_results(*args, **kwds):
    scale_data = kwds['scale_data']
    df = args[1]
    if isinstance(args[0], dict):
        sensor_model_results(args[0], df, scale_data)
    else:
        single_model_results(args[0], df, scale_data)
    
    return df