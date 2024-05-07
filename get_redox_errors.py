import sys
import os.path
import pandas as pd
import pickle
import re
import argparse

import tools.data_parser as dp
import tools.inference_service as inf_serv

__default_model_path = './models/svc_feature_selection_default/'

def verify_file(file_path: str):
    if len(file_path) < 0:
        sys.exit('No file path given. Give file path with argument -f or --file \"python3 get_redox_errors.py -f \/path/to/csv_file.csv\"')
    if not os.path.isfile(file_path):
        sys.exit(f'Could not find file in given path \"{file_path}\"')
    if not file_path.endswith('.csv'):
        sys.exit('Invalid file format, must be .csv file')

def verify_folder(folder_path: str):
    if not os.path.isdir(folder_path):
        sys.exit(f'Could not find folder in given path \"{folder_path}\"')

def get_model_path(model_folder_path: str):
    verify_folder(model_folder_path)
    return model_folder_path
    
def get_parsed_data(data_file_path: str):
    verify_file(data_file_path)
    df = dp.parse_raw_data(data_file_path)

    return df

def get_pickled_model(path: str, name: str):
    model_name = f'{path}/{name}'
    return pickle.load(open(model_name, 'rb'))

def get_models(model_path):
    files = os.listdir(model_path)
    if len(files) == 1:
        return get_pickled_model(model_path, files[0])
    elif len(files) == 5:
        models = dict()
        for file in files:
            m = re.search(r'sensor_\d{1}', file)
            model_key = m.group()
            models[model_key] = get_pickled_model(model_path, file)
        return models
    else:
        sys.exit(f'Invalid amount of models if folder: \"{model_path}\"')

def rename_column_to_original(df: pd.DataFrame, original_names: list[str]):
    name_dict = dict()
    column_names = list(df.columns.array)
    for og_name in original_names:
        current_name = next(name for name in column_names if name in og_name)
        name_dict[current_name] = og_name
    df.rename(columns=name_dict, inplace=True)

def get_model_folders() -> list:
    return os.listdir('./models')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script that uses ML model to find Redox potential errors in given data",
        epilog='''The script can use single model to run inference or multiple models for each sensor separately.
                  The model path is given as folder as the script reads all models in the folder and uses those separately.
                  If sensor specific models are used each model HAS to include \"sensor_<number>\" to verify which model to run on each sensor.
                  Example to run the script with all arguments: python3 get_redox_errors.py -f /path/to/data.csv -m ./models/my_model -s True -o output_file'''
    )

    # define script arguments
    parser.add_argument('-f', '--file', required=True, nargs=1, help='File path of the input data (requires .csv file type). Example: \"-f /path/to/csv_file.csv\"')
    parser.add_argument('-o', '--output', required=True, nargs=1, help='Name for the output file to save as csv. Example: \"-o output_file\"')
    parser.add_argument('-m', '--model', required=False, nargs=1, help='Path to the folder containing the model(s) to use. If not defined default model will be used. Currently only Pickled models are used. Example: \"-m ./models/model_folder/\"')
    parser.add_argument('-s', '--scale', required=False, nargs=1, default='True', choices=['True', 'False'], help='Defines if data is scaled before inference (default: True). Example: \"-s True\"')
    args = parser.parse_args()

    # load arguments data
    file_path = args.file[0]
    output_name = args.output[0]
    model_folder = args.model[0] if args.model else __default_model_path
    scale_data = bool(args.scale[0])

    # get parsed data
    df, original_redox_avg_names = get_parsed_data(file_path)

    # get models
    model_path = get_model_path(model_folder)
    models = get_models(model_path)

    # get results
    results = inf_serv.get_results(models, df, scale_data=True)
    rename_column_to_original(results, original_redox_avg_names)
    # save results
    if not os.path.exists('./Results'):
        os.makedirs('./Results')
    results.to_csv(f'./Results/{output_name}.csv', index=False)