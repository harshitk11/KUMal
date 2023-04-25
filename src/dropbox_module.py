#https://github.com/dropbox/dropbox-sdk-python/blob/main/example/updown.py
"""
Python module to generate parser_info dict. 

parser_info : dict for storing all the parser information for each of the apk folder
                        ## key = Path of apk logcat folder (Contains the apk name)
                        ## value = [Number of logcat files, {logcat_file_1: [avg_freq, num_logcat_lines, time_diff]}, {logcat_file_2: [avg_freq, num_logcat_lines, time_diff]}, ...]

Also has helper functions to analyse the statistics of the parser_info dict. And generates the plot of runtime distributions of malware and benign applications.

"""

import contextlib
import os
import sys
import time
import json
from collections import Counter
from utils import logcat_parser
import traceback
import dropbox
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import requests

def list_folder_extension(dbx, res, rv):
    """
    If the size of the folder is more than 500, then this function will recursively call itself to
    continue to list all the entries in the folder.
    """
    print("Fetching more results")
    res_more = dbx.files_list_folder_continue(res.cursor)

    # Add the entries of the folder to the dict
    rv_more = {}
    for entry in res_more.entries:
        rv_more[entry.name] = entry

    if res_more.has_more:
        rv = list_folder_extension(dbx, res_more, rv)

    # Merge the dict
    rv = {**rv,**rv_more}
    
    return rv


def list_folder(dbx, folder, subfolder):
    """List a folder.
    Return a dict mapping unicode filenames to
    FileMetadata|FolderMetadata entries.
    """
    path = '/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'))
    while '//' in path:
        path = path.replace('//', '/')
    path = path.rstrip('/')
    try:
        with stopwatch('list_folder'):
            res = dbx.files_list_folder(path)
    except (dropbox.exceptions.ApiError,requests.exceptions.ConnectionError) as err:
        print('Folder listing failed for', path, '-- assumed empty:', err)
        return {} # Return empty dict if the folder is empty
    else:
        rv = {}
        for entry in res.entries:
            rv[entry.name] = entry
        
        # Check if res.has_more field is True. If True, then the list of folders is not complete. One call to files_list_folder returns atmost 500 entries.
        # Ref : https://dropbox-sdk-python.readthedocs.io/en/latest/api/dropbox.html
        if res.has_more:
            rv = list_folder_extension(dbx, res, rv)

        return rv 

@contextlib.contextmanager
def stopwatch(message):
    """Context manager to print how long a block of code took."""
    t0 = time.time()
    try:
        yield
    finally:
        t1 = time.time()
        print('Total elapsed time for %s: %.3f' % (message, t1 - t0))

def download(dbx, path, download_path):
    """Download a file.
    Return the bytes of the file, or None if it doesn't exist.
    """
    #  path = '/%s/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'), name)
    while '//' in path:
        path = path.replace('//', '/')
    with stopwatch('download'):
        try:
            dbx.files_download_to_file(download_path, path)
        except dropbox.exceptions.HttpError as err:
            print('*** HTTP error', err)
            return None

def get_folder_list(dbx, dbx_path):
    """
    Generates the list of folder inside the folder specified by dbx_path
    params:
        - dbx : dropbox token
        - dbx_path : Dropbox path of the base folder 

    output:
        - ls_root : dict containing info about all the folders in the folder of interest
                    ## key =  Name of the folder (for a given apk)
                    ## value = Meta-data of the folder [This is not used at all]
        
    """
    # list_folder() returns a dict with key=folder_name
    ls_root=list_folder(dbx,dbx_path,"")
    print(f" - Number of files in the folder ({dbx_path}) = {len(ls_root)}")
    return ls_root    
    

def generate_json_apk_vs_logcat(dbx, ls_root, dbx_path, load_flag, f_save_base_folder):
    """
    Generates json : key="apk folder name" | value = # Logcat files for the apk folder
    params:
        - dbx : dropbox token
        - ls_root : folder list
            ## key =  Name of the folder (for a given apk)
            ## value = Meta-data of the folder [This is not used at all]
        - dbx_path : Dropbox path of the base folder [Used for naming the file]
        - load_flag : If 1, then loads the previously generated json, else generates a new json
        
    Output:
        - logcat_folder_num_runs_dict : {key=apk_folder_path, value=#logcat_files}
    """ 
    dbx_path_ = dbx_path.replace("/","")
    file_loc = os.path.join(f_save_base_folder, f"{dbx_path_}.json")
    
    if load_flag and os.path.isfile(file_loc): # Load the previously generated json
        with open(file_loc,"r") as fp:
            logcat_folder_num_runs_dict = json.load(fp)

    else: # generate and return the new json
        logcat_folder_num_runs_dict={} # Dict - key: subfolder_name, value: Number of logcat files in the subfolder
        for key,value in ls_root.items():
            ## key =  Name of the folder (for a given apk)
            ## value = Meta-data of the folder [This is not used at all]
            
            # Check the number of logcat files inside the subfolder /logcat which is inside the apk folder (stored in key)
            ls_mal_samp=list_folder(dbx, dbx_path+"/"+key,"/logcat")

            # Path of the logcat subfolder for the apk
            sub_folder_name=dbx_path+"/"+key+"/logcat"

            if( bool(ls_mal_samp)): # Check if the subfolder is non-empty before proceeding
                num_runs=0
                for key2,val2 in ls_mal_samp.items():
                    num_runs=num_runs+1

                # Add the entry for this subfolder in the dict
                logcat_folder_num_runs_dict[sub_folder_name] = num_runs
            
            else: # The subfolder doesn't contain any logcat files
                logcat_folder_num_runs_dict[sub_folder_name] = 0 

        # Saving the dictionary once so that we can use it later
        dbx_path_ = dbx_path.replace("/","")
        file_loc = os.path.join(f_save_base_folder, f"{dbx_path_}.json")
        with open(file_loc,"w") as fp:
            json.dump(logcat_folder_num_runs_dict,fp)

    ################################### Printing out the stats #######################################         
    # Post processing the dict () to get the distribution of number of apps vs number of logcat files    
    freq_dist = Counter(logcat_folder_num_runs_dict.values())
    # Sort in the order of number of logcat files
    freq_dist = sorted(freq_dist.items())
    
    for key,value in freq_dist:
        print("Total num of apps with logcat files equal to {} : {}".format(key,value))
    ##################################################################################################   

    return logcat_folder_num_runs_dict

def parse_logcat_files(dbx, dbx_path, dataset_type, benchmark_flag, f_save_base_folder):
    """
    Parses the logcat files and writes the stats of the run in a json file.
    params:
        - dbx : dropbox token
        - dbx_path : Dropbox path of the base folder
        - dataset_type (Used for naming the parser_info logs): Can take one of the following values : ['std_benign', 'std_malware', 'std_vt10_malware', 'cd_year1_benign', 
                                                                'cd_year1_malware', 'cd_year2_benign', 'cd_year2_malware', 'cd_year3_benign', 'cd_year3_malware', 'bench_benign']
        - benchmark_flag : If benchmark flag = True, then we are dealing with benchmarks and we don't have to parse logcat files (since benchmarks always run to completion)
        - f_save_base_folder : Location of the base folder where the logs are loaded from and saved

    Output:
        - parser_info : dict for storing all the parser information for each of the apk folder
                        ## key = Path of apk logcat folder (Contains the apk name)
                        ## value = [Number of logcat files, {logcat_file_1: [avg_freq, num_logcat_lines, time_diff]}, {logcat_file_2: [avg_freq, num_logcat_lines, time_diff]}, ...]
         
    """
    dbx_path_ = dbx_path.replace("/","")
    file_loc = os.path.join(f_save_base_folder, f"{dbx_path_}.json")
    # Load the JSON containing the apk folder name and the number of logcat files for the apk
    with open(file_loc,"r") as fp:
        data=json.load(fp)

    # Create a dict for storing all the parser information for each of the apk folder
    # key = Path of apk logcat folder (Contains the apk name)
    # value = [Number of logcat files, {logcat_file_1: [avg_freq, num_logcat_lines, time_diff]}, {logcat_file_2: [avg_freq, num_logcat_lines, time_diff]}, ...]
    parser_info = {}

    # Iterate through each of the apk file
    for key,value in data.items():
        print(key)
        ## key = Path of the apk logcat subfolder
        ## value = Number of logcat files
        
        # Evaluating whether or not to parse the logcat files
        parse_logcat_flag = False
        if not benchmark_flag: 
            # For the regular and cd dataset we will use all the apks that have logcat files in the range [1,8]
            if value in [1,2,3,4,5,6,7,8]:
                parse_logcat_flag = True
        else:
            # For the benchmark dataset, we will use all the apks with non-zero logcat files
            if value != 0:
                parse_logcat_flag = True
        
        if(parse_logcat_flag): # We will download these folders where there are non-zero logcat files
            
            # Create a dict entry in parser_info for this apk
            parser_info[key] = []
            # Add the number of logcat files for this apk into the dict
            parser_info[key].append(value)

            for key2,val2 in list_folder(dbx,key,"").items(): # Iterate through all the logcat files for the given apk
                # key2 = Logcat file in the logcat subfolder
                # value2 = Meta-data [Not used]

                if not benchmark_flag: # If the apks are not benchmark the parse the logcat files
                    os.system("mkdir -p ../logcat_files/"+key) # Downloaded files are stored in ../logcat_files
                    
                    try:
                        ## download(dbx, path, download_path)
                        download(dbx, key+"/"+key2, "../logcat_files/"+key+"/"+key2)
                        print("--------------- Downloaded ---------------")
                    except:
                        traceback.print_exc()

                        sys.exit()
                        continue
                    
                    # Print for tracking 
                    print(key+"/"+key2)

                    # Get the stats for the logcat file
                    avg_freq = logcat_parser.get_average_frequency(logcat_parser.extract_events("../logcat_files/"+key+"/"+key2))
                    num_logcat_lines = logcat_parser.get_logcat_lines(logcat_parser.extract_events("../logcat_files/"+key+"/"+key2))
                    time_diff = logcat_parser.get_time_difference(logcat_parser.extract_events("../logcat_files/"+key+"/"+key2))
                
                    # Create a dict entry of the stats for this logcat file
                    logcat_stats_entry = {}
                    logcat_stats_entry[key2] = [avg_freq, num_logcat_lines, time_diff]

                    # Append the entry to the parser_info dict
                    parser_info[key].append(logcat_stats_entry)

                    # Remove the apk folder to avoid cluttering
                    os.system("rm -r ../logcat_files"+key.replace('/logcat',''))
                    os.system("rm -r ../logcat_files")

                else: # We have benchmark files. There are no logcat files for benchmarks.
                    # Create a dict entry of the stats for this logcat file
                    logcat_stats_entry = {}
                    # Creating synthetic logcat entry for dependencies in the dataloader
                    logcat_stats_entry[key2] = [100, 100, 100]
                    # Append the entry to the parser_info dict
                    parser_info[key].append(logcat_stats_entry)

            # Dump the file into a JSON for later use
            file_loc = os.path.join(f_save_base_folder, f"parser_info_{dataset_type}.json")
            with open(file_loc,"w") as fp:
                json.dump(parser_info,fp, indent = 4)

def generate_parser_info(dbx, dbx_path, dataset_type, benchmark_flag, load_flag, base_folder_location):
    """
    Generates the parser_info dict for a given dataset_type.
    params:
        - dbx : dropbox token
        - dbx_path : Dropbox path of the base folder
        - dataset_type: Can be one of the following ["std_malware", "std_benign", "cd_malware", "cd_benign"]
        - benchmark_flag : If benchmark flag = True, then we are dealing with benchmarks and we don't have to parse logcat files (since benchmarks always run to completion)
        - load_flag : If load_flag = True, then will load the preexisting json, else will update the list from dropbox
        - base_folder_location : Location of the base folder where all the json files will be stored
    
    Output:
        - parser_info dict
    """
    # Get the list of folders in the base folder specified by dbx_path 
    ls_root = get_folder_list(dbx, dbx_path)
    
    # Get the dict containing the apk folder path and corresponding #logcat_files 
    ## logcat_folder_num_runs_dict : {key=apk_folder_path, value=#logcat_files}
    logcat_folder_num_runs_dict = generate_json_apk_vs_logcat(dbx, ls_root, dbx_path = dbx_path,
                                                             load_flag=load_flag, 
                                                             f_save_base_folder = base_folder_location)

    # Parse the logcat files and write the stats in a json file
    parse_logcat_files(dbx, dbx_path = dbx_path, 
                        dataset_type=dataset_type, benchmark_flag=benchmark_flag, 
                        f_save_base_folder = base_folder_location)

def create_parser_info_for_all_datasets_kumal(dbx, base_folder_location, load_flag):
    """
    Creates parser_info files for all the datasets for the Winter submission.
    params:
        - dbx : dropbox token
        - base_folder_location : location of the base folder where all the logs are stored
        - load_flag : if 1, then will load the previous json containing the apk folder vs num logcat files
    """
    # Generating parser info for STD-dataset, CD_year1_dataset, CD_year2_dataset, CD_year3_dataset
    ## NOTE : Make sure the folder name is preceeded by a backslash 
    std_cd_dataset_info = {
                "std_malware":{"dbx_path":"/logs_std_malware", "app_type":"malware", "dtype":"std_malware"},
                "std_vt10_malware_dev2":{"dbx_path":"/logs_std_vt10_malware_dev2", "app_type":"malware", "dtype":"std_vt10_malware_dev2"},
                "std_benign":{"dbx_path":"/logs_std_benign", "app_type":"benign", "dtype":"std_benign"},
                "std_benign_dev2":{"dbx_path":"/logs_std_benign_dev2", "app_type":"benign", "dtype":"std_benign_dev2"},
                
                "cd_year1_malware":{"dbx_path":"/logs_cd_year1_malware", "app_type":"malware", "dtype":"cd_year1_malware"},
                "cd_year1_benign":{"dbx_path":"/logs_cd_year1_benign", "app_type":"benign", "dtype":"cd_year1_benign"},
                
                "cd_year2_malware":{"dbx_path":"/logs_cd_year2_malware_dev2", "app_type":"malware", "dtype":"cd_year2_malware"},
                "cd_year2_benign":{"dbx_path":"/logs_cd_year2_benign_dev2", "app_type":"benign", "dtype":"cd_year2_benign"},
                
                "cd_year3_malware":{"dbx_path":"/logs_cd_year3_malware", "app_type":"malware", "dtype":"cd_year3_malware"},
                "cd_year3_benign":{"dbx_path":"/logs_cd_year3_benign", "app_type":"benign", "dtype":"cd_year3_benign"}
                }

    for dtype, val in std_cd_dataset_info.items():
        print(f"--------------------- Generating parser_info for {dtype} ---------------------")
        generate_parser_info(dbx=dbx, dbx_path=val["dbx_path"], dataset_type=dtype, 
                            benchmark_flag=False, load_flag = load_flag, base_folder_location = base_folder_location)


#################################### Function to analyse the statistics of information in parser_info dict ####################################
class analyse_parser_info_dict:
    '''
    Contains functions to analyse the statistics and generate runtime plots based on the parser info dicts.
    '''
    parserInfoDict = {
                            "std":{"parserInfoFileName": ["parser_info_std_benign_dev2", "parser_info_std_benign","parser_info_std_malware","parser_info_std_vt10_malware_dev2"],
                                            "apk_type":            ["benign", "benign", "malware", "malware"]},
                            
                            "cdyear1":{"parserInfoFileName": ["parser_info_cd_year1_benign", "parser_info_cd_year1_malware"],
                                                "apk_type":          ["benign", "malware"]},

                            "cdyear2":{"parserInfoFileName": ["parser_info_cd_year2_benign", "parser_info_cd_year2_malware"],
                                                "apk_type":          ["benign", "malware"]},

                            "cdyear3":{"parserInfoFileName": ["parser_info_cd_year3_benign", "parser_info_cd_year3_malware"],
                                                "apk_type":          ["benign", "malware"]}        
                        }

    def __init__(self, base_dir) -> None:
        """
        params:
            - base_dir: path of the src directory of the XMD repository
        """
        # Path where the parser info logs are stored
        self.parser_info_dir = os.path.join(base_dir.replace("/src",""),"res/parser_info_files/kumal")
        # Path where the plots are stored
        self.output_dir_plots = os.path.join(base_dir.replace("/src",""),"plots/dataset_characterization/kumal")

        if not os.path.isdir(self.output_dir_plots):
            os.system(f"mkdir -p {self.output_dir_plots}")

    ##################################### Methods to generate runtime distribution plots #####################################
    def plot_runtime_distribution(self, runtime_per_file, app_type_per_file, dataset_name, save_location):
        '''
        Plots the runtime distribution
        params:
            - runtime_per_file: List containing runtimes of files. (Runtime calculated from logcat files)
            - app_type_per_file: List containing corresponding application type [benign or malware] of files.
            - dataset_name: Dataset ['std','cdyear1', 'cdyear2', 'cdyear3', or 'all'] -> Used in the title of the figure.
            - save_location: Location where the plot is saved
        '''

        # create a dataframe
        d = {'runtime':runtime_per_file, 'apk type':app_type_per_file}
        df = pd.DataFrame(data=d)

        # # calculate percentiles for each group of 'apk type'
        # percentiles = df.groupby('apk type')['runtime'].quantile([0.25, 0.5, 0.75])
        # print(f"percentiles for {dataset_name} : {percentiles}")

        plt.plot()
        palette = ['#5cd65c','#ff5050']
        sns.set(font_scale=1.1)
        sns.set_style("whitegrid",  {"grid.linestyle": "--"})
        ax = sns.histplot(data = df, x='runtime', hue='apk type', binwidth=5, multiple='dodge', shrink=0.8, palette=palette, legend=True)
        plt.xlabel('Runtime (in s)', weight = 'bold')
        plt.ylabel('# of data-collection runs', weight='bold')
        # ax.legend(loc='upper center', ncol =2)
        plt.setp(ax.get_legend().get_title(), weight='bold') # for legend title
        # plt.title(f"Runtime distribution for apks in {dataset_name}")
        plt.tight_layout()
        plt.savefig(save_location)
        plt.close('all')


    def generate_runtime_distribution_plot_per_dataset(self, dataset_name, plot_flag):
        """
        Generates the run-time distribution of the malware and benign samples of a given dataset
        params:
            - dataset_name: The dataset for which the run-time distribution will be generated : ['std', 'cdyear1', 'cdyear2', or 'cdyear3']
            - plot_flag: Boolean flag to determine whether or not to generate the plot

        Output:
            - runtime_per_file, app_type_per_file : Arrays storing the runtime and the corresponding apk type
        """
        
        # Array for storing the runtime duration for the logcat file [for both benign and malware]    
        runtime_per_file = []
        # app type per file
        app_type_per_file = []

        for parserInfoFileName, app_type in zip(analyse_parser_info_dict.parserInfoDict[dataset_name]["parserInfoFileName"], analyse_parser_info_dict.parserInfoDict[dataset_name]["apk_type"]):
            # Load the JSON containing the apk folder name and the number of logcat files for the apk
            with open(os.path.join(self.parser_info_dir,f"{parserInfoFileName}.json"),"r") as fp:
                data=json.load(fp)
                
            for key,value in data.items():
                # key = Path of apk logcat folder (Contains the apk name)
                # value = [Number of logcat files, {logcat_file_1: [avg_freq, num_logcat_lines, time_diff]}, {logcat_file_2: [avg_freq, num_logcat_lines, time_diff]}, ...]
            
                for ind in range(value[0]): # Value[0] number of logcat files for each apk. Each logcat file has its own dict.
                    i = ind + 1 # For indexing into the corresponding dict in the list.
                    for key_,value_ in value[i].items():
                        # key_ = Name of the logcat file
                        # Value = [avg_freq, num_logcat_lines, time_diff]
                        runtime_per_file.append(value_[2])
                        app_type_per_file.append(app_type)

        if plot_flag:
            self.plot_runtime_distribution(runtime_per_file=runtime_per_file,
                                            app_type_per_file=app_type_per_file,
                                            dataset_name=dataset_name,
                                            save_location=os.path.join(self.output_dir_plots,f"runtime_distribution_malware_benign_{dataset_name}_dataset.png"))

        return runtime_per_file, app_type_per_file

    def generate_runtime_distribution_plot_all_datasets(self, plot_flag):
        """
        Generates the run-time distribution of the malware and benign samples of all the datsets combined: {std-dataset and cd-dataset}
        """
    
        runtime_per_file_std, app_type_per_file_std = self.generate_runtime_distribution_plot_per_dataset(dataset_name='std', plot_flag=False)
        runtime_per_file_cd1, app_type_per_file_cd1 = self.generate_runtime_distribution_plot_per_dataset(dataset_name='cdyear1', plot_flag=False)
        runtime_per_file_cd2, app_type_per_file_cd2 = self.generate_runtime_distribution_plot_per_dataset(dataset_name='cdyear2', plot_flag=False)
        runtime_per_file_cd3, app_type_per_file_cd3 = self.generate_runtime_distribution_plot_per_dataset(dataset_name='cdyear3', plot_flag=False)

        runtime_per_file = runtime_per_file_std+runtime_per_file_cd1+runtime_per_file_cd2+runtime_per_file_cd3
        app_type_per_file = app_type_per_file_std+app_type_per_file_cd1+app_type_per_file_cd2+app_type_per_file_cd3

        if plot_flag:
            self.plot_runtime_distribution(runtime_per_file=runtime_per_file,
                                            app_type_per_file=app_type_per_file,
                                            dataset_name='all',
                                            save_location=os.path.join(self.output_dir_plots,f"runtime_distribution_malware_benign_all_dataset.pdf"))
    ########################################################################################################################

#######################################################################################################################################################

def main():
    # Current directory path [the folder in which the script is stored]
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # Get the dropbox api key
    with open("/data/hkumar64/projects/arm-telemetry/xmd/src/dropbox_api_key") as f:
        access_token = f.readlines()[0]

    # Authenticate with Dropbox
    print('Authenticating with Dropbox...')
    dbx = dropbox.Dropbox(access_token)
    print('...authenticated with Dropbox owned by ' + dbx.users_get_current_account().name.display_name)

    # Location of the base folder where all the parser info logs will be stored and loaded from 
    base_folder_location = os.path.join(dir_path.replace("/src",""),"res/parser_info_files/kumal")
    
    if not os.path.isdir(base_folder_location):
        os.system(f"mkdir -p {base_folder_location}")

    # Generating parser_info for all the datasets [STD, CD, and BENCH dataset]
    # create_parser_info_for_all_datasets_kumal(dbx, base_folder_location=base_folder_location, load_flag=0)
    # exit()
    
    # ################################################### Analyze the parser_info dicts ############################################################
    # Generate the runtime distribution plots
    dataset_characterization = analyse_parser_info_dict(base_dir=dir_path)
    dataset_characterization.generate_runtime_distribution_plot_per_dataset(dataset_name = 'cdyear1', plot_flag=True)
    dataset_characterization.generate_runtime_distribution_plot_per_dataset(dataset_name = 'cdyear2', plot_flag=True)
    dataset_characterization.generate_runtime_distribution_plot_per_dataset(dataset_name = 'cdyear3', plot_flag=True)
    dataset_characterization.generate_runtime_distribution_plot_per_dataset(dataset_name = 'std', plot_flag=True)
    dataset_characterization.generate_runtime_distribution_plot_all_datasets(plot_flag=True)
    # ##############################################################################################################################################


if(__name__=="__main__"):
    main()


    