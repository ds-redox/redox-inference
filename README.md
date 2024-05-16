# redox-inference
Script for predicting redox errors

# Clone the repository to own machine

[How to clone repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)

[Using ssh with GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

It is recommended to use virtual environment so that the required python packages are the same. [How to set up python virtaul environment](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/)

Afther these are ready change directory to the project and install required Python packages with `pip install -r requirements.txt`.

# Components & description

The repository containts the main script `get_redox_errors.py` which is called by the user

`tools` folder contains helper scripts to parse the given data, adds additional features, and logic to run the inference.

`models` folder contains all of the models that the script can use. In order to add new models, create a new folder with descriptive name and paste a **pickle** serialized model inside the folder. For sensors specific models add all 5 pickled models inside the folder. **NOTE** sensor specific models need to have the word `sensor_<number>` in the pickle file name to work correctly.

If sensors specific models are used the script gets predictions for each sensor model and combines all sensor findigs as one redox_error_flag by using logical OR on each result.

# How to use

Minimum way to run the script `python3 get_redox_errors.py -f /path/to/file.csv -o my_results`. All of the script options are defined below.

To get help, run the following command `python3 get_redox_errors.py --help`. This will output description of the script and guide how to use it.

### Options:

-f (--file) &nbsp;&nbsp; (Required option). This option defines the path to the file that is fed to the model. Given file type must be csv.

-o (--output) &nbsp;&nbsp; (Required option). This options defines the name of the output file that contains the redox error findings.

-m (--model) &nbsp;&nbsp; (Optional). This option defines the model used for the inference. By default it uses sensor specific SVM models.

-s (--scale) &nbsp;&nbsp; (Optional). This option defines if the data is scaled before the inference. The default value is True
