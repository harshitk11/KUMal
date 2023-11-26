# Repository for the codebase and the dataset of KUMal.
## Dataset-collection framework.
For this study, we developed an automated bare-metal sandbox for large-scale data-collection of hardware telemetry logs. The framework supports Android-OS Client, and has been designed specifically for a Google Pixel-3 device. The Host PC where the main [orchestrator](/baremetal_data_collection_framework/orchestrator.py) runs is a Linux-OS based PC. To prevent local-storage overhead, the sandbox is closely integrated with cloud (Dropbox). After every completion of data-collection for every application (totalling 8 iterations of runs with a duration of 90 seconds per iteration), the logs are pushed to Dropbox saving local storage space. Please see [readme](/baremetal_data_collection_framework/README.md) of the bare-metal data collection framework for more info.

## Usage
To create the conda environment use the requirements.yml file (`conda env create -f requirements.yml -p conda-env`). Make sure [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) is installed on your system.

Steps to generate the dataset:
1. Create json with info about the different runs: The first step is to parse the logcat files and create json file which has the apk folder location and the different attributes of the logcat files based on which filtering will be performed.
    - To do this we run `python dropbox_module.py`
        - This will generate json files (parser_info_*) in the directory `res/parser_info_files` which will be used for filtering out the logs of application runs with not sufficient runtimes.

2. Filter out the apk runs using filter values for run-time, and download the shortlisted runs from Dropbox.
    - To do this we run `python create_dataset.py`
        - Specifically, we run the `generate_dataset()` method of the `dataset_generator` class. The filter values are passed as arguments when an instance of the `dataset_generator` class is created. 


## Functionalities provided by each script in [source directory](/src)
- `utils.py`: Contains all the basic utilities used by different modules.
    1. `logcat_parser`: Class to parse the logcat file to extract the following information from it: (1) Number of lines,  (2) Timestamp difference to see how long the application executes, and (3) Rate at which logcat events are happening.
    2. `Config` : Class to handle the config files.
    3. `malware_label_generator` : Contains helper functions to generate VT reports for AVClass.


- `dropbox_module.py`:-
    1. `create_parser_info_for_all_datasets()`: Python module to generate parser_info dict for all the datasets: `std-dataset`, `cd-dataset`, and `bench-dataset`.

        parser_info : dict for storing all the parsed information for each of the apk folder
                    ## key = Path of apk logcat folder (Contains the apk name), 
                    ## value = [Number of logcat files, {logcat_file_1: [avg_freq, num_logcat_lines, time_diff]}, {logcat_file_2: [avg_freq, num_logcat_lines, time_diff]}, ...]

    2. `analyse_parser_info_dict`: Class to analyse the statistics and generate runtime plots based on the parser info dicts. `generate_runtime_distribution_plot_all_datasets()` generates the runtime distribution plot of all the apk runs in all the datasets.
    
- `create_raw_dataset.py`: Python script to create the dataset containing the raw GLOBL and HPC files. 
    1. `dataset_generator_downloader` : Class that filters out the runs (based on the filter) and downloads the dataset from Dropbox.
    2. `arm_telemetry_data` : Class containing the dataset object. Reads, parses, and creates a batch of files.
    3. `custom_collator` : Class containing the collator for the dataloader. Contains all the feature engineering steps for the GLOBL and HPC channels.
    4. `dataset_split_generator` : Generates the dataset splits for all the classification tasks: DVFS individual, DVFS Fusion, HPC individual, and HPC-DVFS Fusion.
    5. `get_dataloader()` : Returns the dataloader objects for the different partitions.

- `create_featEngineer_dataset.py`: Python script to generate the feature engineered dataset from the raw dataset for different experimental parameters (e.g., logcat_runtime_threshold, truncated_duration)
    1. `dataloader_generator` : Class that contains all the helper methods for generating the dataloader for all the classification tasks.
    2. `feature_engineered_dataset` : Class containing all the methods for creating the feature engineered datasets for HPC and GLOBL channels.
    

