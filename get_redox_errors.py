import sys
import os.path

import tools.data_parser as dp

def verify_file(file_path: str):
    if not os.path.isfile(file_path):
        sys.exit(f'Could not find file in given path \"{file_path}\"')
    if not file_path.endswith('.csv'):
        sys.exit('Invalid file format, must be .csv file')

def get_path_argument():
    try:
        path_arg = sys.argv[1]
        return path_arg
    except:
        sys.exit('No file path given. Give file path as first agrument \"python3 get_redox_errors.py \/path/to/csv_file.csv\"')

if __name__ == '__main__':
    file_path = get_path_argument()
    verify_file(file_path)
    df = dp.get_data(file_path)
    print(list(df.columns.array))
    print(df.head(1))

    # file = pd.read_csv()
    # preprocess()
    # inference()