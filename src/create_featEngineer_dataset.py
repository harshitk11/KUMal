"""
Python script to generate the feature engineered dataset from the raw dataset 
for different experimental parameters (e.g., logcat_runtime_threshold, truncated_duration)
"""
import argparse
import datetime
import torch
from utils import Config
from create_raw_dataset import dataset_generator_downloader, dataset_split_generator, custom_collator, get_dataloader
import os
import shutil
import numpy as np
import pickle
import json
import tqdm

# Used for storing the latest list of local paths of files that should be downloaded for creating the dataset.
timeStampCandidateLocalPathDict = None

def get_args(xmd_base_folder):
    """
    Reads the config file and returns the config parameters.
    params:
        - xmd_base_folder: Location of xmd's base folder
    Output:

    """
    parser = argparse.ArgumentParser(description="XMD : Late-stage fusion.")
    # Location of the default and the update config files.
    parser.add_argument('-config_default', type=str, dest='config_default', default=os.path.join(xmd_base_folder,'config','default_config.yaml'), help="File containing the default experimental parameters.")
    parser.add_argument('-config', type=str, dest='config', default=os.path.join(xmd_base_folder,'config','update_config.yaml'), help="File containing the experimental parameters to update.")
    opt = parser.parse_args()

    # Create a config object. Initialize the default config parameters from the default config file.
    cfg = Config(file=opt.config_default)

    # Update the default config parameters with the parameters present in the update config file.
    cfg.update(updatefile=opt.config)

    # Get the config parameters in args [args is an easydict]
    args = cfg.get_config()
    args['default_config_file'] = opt.config_default
    args['update_config_file'] = opt.config

    # Timestamp for timestamping the logs of a run
    timestamp = str(datetime.datetime.now()).replace(':', '-').replace(' ', '_')
    args.timestamp = timestamp

    return cfg, args

class dataloader_generator:
    """
    Contains all the helper methods for generating the dataloader for all the classification tasks.
    """
    # Used to identify the partitions required for the different classification tasks for the different datasets : std-dataset, cd-dataset, bench-dataset
    partition_activation_flags = {
                                    "std-dataset":{"train":True, "test":True},
                                    "cd-dataset":{"train":False, "test":True},           
                                }

    @staticmethod
    def get_dataset_type_and_partition_dist(dataset_name):
        """
        Returns the dataset type and the partition dist for the dataset_split_generator.
        params:
            - dataset_name : Name of dataset. Can take one of the following values: {"std-dataset","bench-dataset","cd-dataset",
                                                                                    "cdyear1-dataset","cdyear2-dataset","cdyear3-dataset"}
                                                                                    
        Output:
            - dsGen_dataset_type : Type of dataset. Can take one of the following values: {"std-dataset","bench-dataset","cd-dataset"}
            - dsGem_partition_dist : Split of the partition [num_train_%, num_trainSG_%, num_test_%]
        """
        if dataset_name == "std-dataset":
            # This dataset is used for training the base-classifiers and the second stage model
            dsGen_dataset_type = "std-dataset"
            dsGem_partition_dist = [0.70,0.30]
        
        elif (dataset_name == "cd-dataset") or (dataset_name == "cdyear1-dataset") or (dataset_name == "cdyear2-dataset") or (dataset_name == "cdyear3-dataset"):
            # This dataset is used for training and testing the base-classifiers. The goal of the dataset is to establish the non-determinism in the GLOBL channels.
            dsGen_dataset_type = "cd-dataset"
            dsGem_partition_dist = [0,1]
        
        else:
            raise ValueError("[main_worker] Incorrect dataset type specified for the dataset split generator.")

        return dsGen_dataset_type, dsGem_partition_dist

    @staticmethod
    def prepare_dataloader(partition_dict, labels, dataset_type, args):
        """ 
        Configure the dataloader. Based on the dataset type and the classification task of interest,
        this will return dataloaders for the tasks: "train","test".

        params: 
            - partition_dict : {'train' : [file_path1, file_path2, ..],
                                'test' : [file_path1, file_path2]}
            - labels : {file_path1 : 1, file_path2: 0, ...}
            - dataset_type : Type of dataset. Can take one of the following values: {"std-dataset","cd-dataset"}
            - args :  arguments from the config file

        Output: 
            - train and test dataloader objects depending on the dataset_type
        """

        print(f'[Info] Fetching dataloader objects for {dataset_type}')
    
        # Find the partitions that will be required for this dataset. required_partitions = {"train":T or F, "test":T or F}
        required_partitions = dataloader_generator.partition_activation_flags[dataset_type]
        
        # Intitialize the custom collator
        custom_collate_fn = custom_collator(args=args)
 
        # Get the dataloader object : # get_dataloader() returns an object that is returned by torch.utils.data.DataLoader
        trainloader, testloader = get_dataloader(args=args, 
                                                partition = partition_dict, 
                                                labels = labels, 
                                                custom_collate_fn =custom_collate_fn,
                                                required_partitions=required_partitions)

        return {'trainloader': trainloader, 'testloader': testloader}
    
    @staticmethod
    def get_dataloader_for_all_classification_tasks(HPC_partitions_and_labels_for_all_rn, dataset_type, args):
        """
        Generates the dataloader for all the classification tasks.
        NOTE: args.truncated_duration is used to truncate the timeseries. If you want truncated time series, then args.truncated_duration
              needs to be changed before calling this method.

        params:
            - HPC_partitions_and_labels_for_all_rn: [HPC_partition_for_HPC_individual,HPC_partition_labels_for_HPC_individual]
                        
                        -> HPC_partition_for_HPC_individual: [HPC_partition_for_HPC_individual_with_rn1, HPC_partition_for_HPC_individual_with_rn2, ...rn3, ...rn4]
                        -> HPC_partition_labels_for_HPC_individual: [HPC_partition_labels_for_HPC_individual_with_rn1, HPC_partition_labels_for_HPC_individual_with_rn2, ...rn3, ...rn4]
                               
                                partition -> {'train' : [file_path1, file_path2, ..],
                                                'test' : [file_path1, file_path2]}

                                labels -> {file_path1 : 0, file_path2: 1, ...}        [Benigns have label 0 and Malware have label 1]
            
            - dataset_type: Can take one of the following values {'std-dataset','bench-dataset','cd-dataset'}. 
                            Based on the dataset type we will activate different partitions ("train", "trainSG", "test") for the different classification tasks.

            - args: easydict storing the arguments for the experiment

        Output:
            - dataloader_all_rn =
                                    {        
                                        'rn1':{'trainloader': trainloader, 'testloader': testloader},
                                        'rn2':{'trainloader': trainloader, 'testloader': testloader},
                                        'rn3':{'trainloader': trainloader, 'testloader': testloader},
                                        'rn4':{'trainloader': trainloader, 'testloader': testloader}
                                    }
                    NOTE: If the dataloader for a particular task is not required, then it is None.
        """
        # For selecting the dataset partition and labels
        select_dataset = {
                            'rn1':{'partition':HPC_partitions_and_labels_for_all_rn[0][0],'label':HPC_partitions_and_labels_for_all_rn[1][0]},
                            'rn2':{'partition':HPC_partitions_and_labels_for_all_rn[0][1],'label':HPC_partitions_and_labels_for_all_rn[1][1]},
                            'rn3':{'partition':HPC_partitions_and_labels_for_all_rn[0][2],'label':HPC_partitions_and_labels_for_all_rn[1][2]},
                            'rn4':{'partition':HPC_partitions_and_labels_for_all_rn[0][3],'label':HPC_partitions_and_labels_for_all_rn[1][3]}
                        }
        
        print(f"\t************ Details of dataloader generation for dataset name : {dataset_type} ***************")
        # Get the dataloader for all the classification toi
        dataloader_all_rn = {}
        for rnBin, partitionDetails in select_dataset.items(): 
            dataloader_all_rn[rnBin] = dataloader_generator.prepare_dataloader(partition_dict = partitionDetails['partition'], 
                                                                                        labels = partitionDetails['label'], 
                                                                                        dataset_type = dataset_type, 
                                                                                        args = args)
            # if partitionDetails['partition'] is not None:
            #     print(clf_toi,rnBin)
            #     for partitionName, partitionDet in partitionDetails['partition'].items():
            #         try:
            #             print(f"{partitionName} -> {len(partitionDet)}")
            #         except:
            #             print(f"{partitionName} -> {None}")

        # ################################################################ Testing the supervised learning dataloader ################################################################
        # iter_loader = iter(dataloader_all_rn['DVFS_fusion']['all']['testloader'])
        # batch_spec_tensor, labels, f_paths = next(iter_loader)
        # f_paths = "\n - ".join(f_paths)
        # print(f"- Shape of batch tensor (B,N_ch,reduced_feature_size) : {batch_spec_tensor.shape}")
        # print(f"- Batch labels : {labels}")
        # print(f"- File Paths : {f_paths}")
        # exit()

        # iter_loader = iter(dataloader_all_rn['HPC_individual']['rn1']['trainloader'])
        # batch_spec_tensor, labels, f_paths = next(iter_loader)
        # f_paths = "\n - ".join(f_paths)
        # print(f"- Shape of batch tensor (B,N_ch,reduced_feature_size) : {batch_spec_tensor.shape}")
        # print(f"- Batch labels : {labels}")
        # print(f"- File Paths : {f_paths}")
        # exit()

        # # Testing the alignment of DVFS and HPC for HPC-DVFS fusion
        # # HPC
        # iter_testloader_hpc = iter(dataloader_all_rn['HPC_partition_for_HPC_DVFS_fusion']['rn2']['trainSGloader'])
        # batch_spec_tensor_hpc, labels_hpc, f_paths_hpc = next(iter_testloader_hpc)
        # # DVFS
        # iter_testloader_dvfs = iter(dataloader_all_rn['DVFS_partition_for_HPC_DVFS_fusion']['rn2']['trainSGloader'])
        # batch_spec_tensor_dvfs, labels_dvfs, f_paths_dvfs = next(iter_testloader_dvfs)
        # for i,j in zip(f_paths_dvfs,f_paths_hpc):
        #     print(f"-- {i,j}")
        # exit()
        ###############################################################################################################################################################################
        return dataloader_all_rn
        
class feature_engineered_dataset:
    """
    Class containing all the methods for creating the feature engineered datasets for HPC and GLOBL channels.
    """
    def __init__(self, args, HPC_partitions_and_labels_for_all_rn, dataset_type, results_path) -> None:
        """
        params:
            - HPC_partitions_and_labels_for_all_rn: [HPC_partition_for_HPC_individual,HPC_partition_labels_for_HPC_individual]
                        
                        -> HPC_partition_for_HPC_individual: [HPC_partition_for_HPC_individual_with_rn1, HPC_partition_for_HPC_individual_with_rn2, ...rn3, ...rn4]
                        -> HPC_partition_labels_for_HPC_individual: [HPC_partition_labels_for_HPC_individual_with_rn1, HPC_partition_labels_for_HPC_individual_with_rn2, ...rn3, ...rn4]
                               
                                partition -> {'train' : [file_path1, file_path2, ..],
                                                'test' : [file_path1, file_path2]}

                                labels -> {file_path1 : 0, file_path2: 1, ...}        [Benigns have label 0 and Malware have label 1]
                
            - dataset_type: Can take one of the following values {'std-dataset','bench-dataset','cd-dataset'}. 
                            Based on the dataset type we will activate different partitions ("train", "test") for the different classification tasks.
            - results_path: Location of the base folder where all the feature engineered datasets are stored.    
        """
        self.args = args
        self.HPC_partitions_and_labels_for_all_rn = HPC_partitions_and_labels_for_all_rn
        self.dataset_type = dataset_type
        self.results_path = results_path
        
        # List of candidate truncated durations
        self.rtime_list = [i for i in range(args.step_size_truncated_duration, args.collected_duration+args.step_size_truncated_duration, args.step_size_truncated_duration)]

        # logcat_filter_rtime: Filter runtime used when filtering and downloading the dataset (in s) [Used for naming the output file]
        self.logcat_filter_rtime_threshold = args.runtime_per_file

        
    @staticmethod
    def dataset_download_driver(args, xmd_base_folder_location, numApp_info_dict, FolderName_featureEngineeredDatasetDetails):
        """
        Downloads the dataset if it's not already downloaded. Trims the dataset (based on the logcat runtime filter) if downloaded.
        
        params:
            - args : Uses args.collected_duration to calculate the list of logcat runtime thresholds
            - xmd_base_folder_location: Base folder of xmd's source code
        """
        # Folder where the dataset is downloaded
        datasetDownloadLoc = os.path.join(args.dataset_base_location, args.dataset_name)
        
        global timeStampCandidateLocalPathDict
        dataset_generator_instance = dataset_generator_downloader(filter_values= [args.runtime_per_file, args.num_logcat_lines_per_file, args.freq_logcat_event_per_file], 
                                                                        dataset_name=args.dataset_name,
                                                                        base_download_dir=args.dataset_base_location)

        # Location where timestamp dict is stored
        timeStampCandidateLocalPathDict_saveLocation = os.path.join(xmd_base_folder_location,"res",FolderName_featureEngineeredDatasetDetails,f"timeStampCandidateLocalPathDict_{args.dataset_name}.pkl")
        
        # ######################################## For debugging ########################################    
        # _,_,candidateLocalPathDict = dataset_generator_instance.generate_dataset_winter(download_file_flag=False, num_download_threads=args.num_download_threads)
        #     # Update the timestamp list
        # timeStampCandidateLocalPathDict = candidateLocalPathDict
        # ###############################################################################################

        # If the dataset is not downloaded, then download the dataset
        if not os.path.isdir(datasetDownloadLoc):
            _,_,candidateLocalPathDict = dataset_generator_instance.generate_dataset_winter(download_file_flag=args.dataset_download_flag, num_download_threads=args.num_download_threads)
            # Update the timestamp list
            timeStampCandidateLocalPathDict = candidateLocalPathDict
            # Save the timestamp list
            with open(timeStampCandidateLocalPathDict_saveLocation, 'wb') as fp:
                pickle.dump(timeStampCandidateLocalPathDict, fp)

        # Checkpointing : if timeStampCandidateLocalPathDict is None, then you need to load the previously saved list
        elif timeStampCandidateLocalPathDict is None:
            # Load the saved timeStampCandidateLocalPathDict
            with open(timeStampCandidateLocalPathDict_saveLocation, 'rb') as fp:
                timeStampCandidateLocalPathDict = pickle.load(fp)
            
        # If the dataset is downloaded, then generate list of files to delete from the downloaded dataset.
        elif os.path.isdir(datasetDownloadLoc):
            print("***************** Dataset already downloaded. Trimming the dataset. *****************")
            # First generate the list of candidate local paths
            _,_,candidateLocalPathDict = dataset_generator_instance.generate_dataset_winter(download_file_flag=False)
            
            # Based on the list and the previous timestamplist, generate the list of files to be deleted from the downloaded dataset
            deleteFilePaths = {}
            for pathLabel,pathList in candidateLocalPathDict.items():
                deleteFilePaths[pathLabel] = [x for x in timeStampCandidateLocalPathDict[pathLabel] if x not in pathList]

            # Delete the files.
            for pathLabel,pathList in deleteFilePaths.items():
                for fpath in pathList:
                    # Delete the file if it exists
                    try:
                        os.remove(fpath)
                        print(f" - Deleted the file : {fpath}")
                    except OSError:
                        print(f" - File not found to delete : {fpath}")

            # Update the timestamp list with the new candidateLocalPathDict
            timeStampCandidateLocalPathDict = candidateLocalPathDict
            # Save the timestamp list
            with open(timeStampCandidateLocalPathDict_saveLocation, 'wb') as fp:
                pickle.dump(timeStampCandidateLocalPathDict, fp)

        ############################ Log the info about this dataset ############################
        # Count the number of apks from the shortlisted files (This is the number of apks post logcat filter)
        num_benign,num_malware = dataset_generator_instance.count_number_of_apks()
        numApp_info_dict[args.dataset_name][args.runtime_per_file] = {"NumBenignAPK":num_benign, "NumMalwareAPK":num_malware, "logcatRuntimeThreshold": args.runtime_per_file}

        PATH_dataset_info_num_apk = os.path.join(xmd_base_folder_location,"res",FolderName_featureEngineeredDatasetDetails, "dataset_info_num_apk.json")
        with open(PATH_dataset_info_num_apk,'w') as fp:
            json.dump(numApp_info_dict,fp, indent=2)
        #########################################################################################
        
        return numApp_info_dict


    @staticmethod
    def generate_feature_engineered_dataset(args, xmd_base_folder_location, featEngineerDatasetFolderName, FolderName_featureEngineeredDatasetDetails):
        """
        Function to generate the feature engineered dataset by doing a sweep over the two parameters: logcat-runtime_per_file, truncated_duration 
        A new dataset is created for each tuple (logcat-runtime_per_file, truncated_duration).

        Writes the details of the generated dataset in the runs directory.

        High-level pseudo code:
            Download the raw dataset for 0 logcat runtime threshold
            Generate the feature engineered dataset for all truncated durations
            For each logcat runtime threshold:
                Trim the downloaded raw dataset.
                Generate the feature engineered dataset for all truncated durations
        
        params:
            - args : Uses args.collected_duration to calculate the list of logcat runtime thresholds
            - xmd_base_folder_location: Base folder of xmd's source code
            - featEngineerDatasetFolderName: Name of the base folder for storing the feature engineered dataset.
        """
        if not os.path.isdir(os.path.join(xmd_base_folder_location,"res",FolderName_featureEngineeredDatasetDetails)):
            os.mkdir(os.path.join(xmd_base_folder_location,"res",FolderName_featureEngineeredDatasetDetails))
        
        # Generate a list of the logcat-runtime_per_file values i.e. the iterations that we are downloading has the apks running atleast logcat-runtime_per_file seconds.
        logcat_rtimeThreshold_list = [i for i in range(0, args.collected_duration, args.step_size_logcat_runtimeThreshold)]

        # Load the json file containing the info about the number of benign and malware apks in the filtered dataset
        PATH_dataset_info_num_apk = os.path.join(xmd_base_folder_location,"res",FolderName_featureEngineeredDatasetDetails, "dataset_info_num_apk.json")
        if os.path.isfile(PATH_dataset_info_num_apk):
            with open(PATH_dataset_info_num_apk,'r') as fp:
                numApp_info_dict = json.load(fp)
            if args.dataset_name not in numApp_info_dict.keys():
                numApp_info_dict[args.dataset_name] = {}
        else:
            numApp_info_dict = {args.dataset_name:{}}
        
        for logcatRtimeThreshold in logcat_rtimeThreshold_list:
            # Set the runtime threshold which is used by dataset_generator_downloader
            args.runtime_per_file = logcatRtimeThreshold
            
            # Download the raw-dataset for the specific value of the logcat filter. Trim the dataset if an older version exists.
            if args.dataset_download_flag:
                numApp_info_dict = feature_engineered_dataset.dataset_download_driver(args=args, 
                                                                    xmd_base_folder_location=xmd_base_folder_location, 
                                                                    numApp_info_dict = numApp_info_dict,
                                                                    FolderName_featureEngineeredDatasetDetails=FolderName_featureEngineeredDatasetDetails)

            # Get the dataset type and the partition dist for the dataset split generator
            dsGen_dataset_type, dsGem_partition_dist = dataloader_generator.get_dataset_type_and_partition_dist(dataset_name= args.dataset_name)
            
            # Generate the dataset splits (partition dict and the labels dict)
            dataset_base_location = os.path.join(args.dataset_base_location, args.dataset_name)
            dsGen = dataset_split_generator(seed=args.seed, 
                                            partition_dist=dsGem_partition_dist)
            HPC_partitions_and_labels_for_all_rn = dsGen.create_HPC_partitions_and_labels_for_all_rn(base_location=dataset_base_location)

            ################################################## Generate feature engineered dataset for all truncated durations ##################################################
            featEngDatsetBasePath = os.path.join(xmd_base_folder_location,"data",featEngineerDatasetFolderName,args.dataset_name)
            featEngineeringDriver = feature_engineered_dataset(args=args, 
                                                        HPC_partitions_and_labels_for_all_rn = HPC_partitions_and_labels_for_all_rn, 
                                                        dataset_type = dsGen_dataset_type, 
                                                        results_path = featEngDatsetBasePath)
            featEngineeringDriver.generate_feature_engineered_dataset_per_logcat_filter()
            #####################################################################################################################################################################
            

    def generate_feature_engineered_dataset_per_logcat_filter(self):
        """
        Method to generate the feature engineered dataset for all the classification tasks for a given logcat filter-value.
        """
        for rtime in self.rtime_list:
            print(f"\t----------- Generating feature engineered dataset for truncated duration : {rtime} -----------")
            self.generate_feature_engineered_dataset_per_rtime_per_logcat_filter(truncated_duration=rtime)
            # exit()

    def generate_feature_engineered_dataset_per_rtime_per_logcat_filter(self, truncated_duration):
        """
        Method to generate the feature engineered dataset for all the classification tasks for a given filter-value and truncated-duration.
        
        params:
            - truncated_duration: time to which you want to trim the time series (in s)
        """
        # Set the truncated duration in the args
        self.args.truncated_duration = truncated_duration
        # Generate the dataloaders for all the classification tasks
        dataloader_all_rn = dataloader_generator.get_dataloader_for_all_classification_tasks(HPC_partitions_and_labels_for_all_rn = self.HPC_partitions_and_labels_for_all_rn, 
                                                                                        dataset_type=self.dataset_type, 
                                                                                        args=self.args)
    
        # Generate feature engineered vectors
        for rnBin, dataloaderDict in dataloader_all_rn.items():
            for dataloaderName, dataloaderX in dataloaderDict.items():
                if dataloaderX is not None:
                    # Generate the feature engineered dataset
                    splitName = dataloaderName.replace("loader","")
                    print(f"Generating channel bins for : {rnBin, splitName}")
                    try:
                        print(f" - Number of samples : {len(dataloaderX.dataset)} | Number of batches: {len(dataloaderX)}")
                        # continue
                    except:
                        print("error in printing length")

                    self.create_channel_bins(dataloaderX=dataloaderX,
                                            truncated_duration= self.args.truncated_duration,
                                            rnBin=rnBin,
                                            split_type=splitName) 
                       
    def create_channel_bins(self, dataloaderX, truncated_duration, rnBin, split_type):
        """
        Creates the dataset (post feature engineering) for the different classification tasks. 
        Writes the dataset, corresponding labels, and file paths to npy files.
        
        params:
            - dataloaderX : dataloader for the training dataset that returns (batch_tensor, batch_labels, batch_paths) 
                            -batch_tensor (batch of iterations) : Shape - B, N_channels, feature_size
                            -batch_labels (label for each iteration in the batch) : Shape - B
                            -batch_paths (path of the file for each iteration in the batch) : Shape - B
            
            ----------------------------- Used for labelling the output files ----------------------------- 
            - self.logcat_filter_rtime_threshold
            - truncated_duration: time to which you want to trim the time series (in s)           
            - rnBin : "rn1","rn2","rn3","rn4"
            - split_type : "train" or "test" 
            -----------------------------------------------------------------------------------------------
            
        Output:
            Iterate over all the batches in the dataset and separate each of the channels into their respective bins
            - channel_bins : channel_bins[i][j] : stores the reduced feature of the jth iteration of the ith channel [Shape: N_ch, N_samples, feature_size]
            - labels       : labels[j] : stores the labels of the jth iteration
            - f_paths      : f_paths[j] : stores the file path (contains the file name) of the jth iteration

        Returns the paths of the files where channel_bins, labels, f_paths are stored, i.e., 
        {"path_channel_bins": ... , "path_labels": ... , "path_f_paths": ... }
        """
        # Determine the number of channels [=15 for GLOBL, and =1 for HPC]
        test_channel_bins,_,_ = next(iter(dataloaderX))
        _,num_channel_bins,_ = test_channel_bins.shape
         
        channel_bins = [[] for _ in range(num_channel_bins)] # Stores the reduced_feature of each channel
        labels = [] # Stores the labels of the corresponding index in the channel_bins
        f_paths = [] # Stores the file paths of the corresponding index in the channel_bins

        iterx = iter(dataloaderX)
        for batch_idx,(batch_tensor, batch_labels, batch_paths) in enumerate(tqdm.tqdm(dataloaderX)):
            # Get the dimensions of the batch tensor
            B,N_ch,ft_size = batch_tensor.shape

            # print(f"  [{batch_idx}] Shape of batch (B,N_ch,reduced_feature_size) : {batch_tensor.shape}")
            
            # Split the batch into individual iterations
            for iter_idx,iterx in enumerate(torch.unbind(batch_tensor,dim=0)):
                # print(f"Shape of iteration tensor (Nch, reduced_feature_size): {iter_idx,iterx.shape}")

                # Add the label for this iteration into the labels array
                labels.append(batch_labels[iter_idx].item())

                # Add the file path for this iteration into the f_paths array
                f_paths.append(batch_paths[iter_idx])

                # Split the iteration into channels and store the channels into their respective bins
                for chn_idx,chn in enumerate(torch.unbind(iterx,dim=0)):
                    # print(f"Shape of channel tensor (reduced_feature_size): {chn_idx,chn.shape}")
                    
                    # create a new tensor that doesn't share the same underlying address
                    chn = chn.clone()
                    
                    # Add it to the channel bin
                    channel_bins[chn_idx].append(chn.numpy())
                    # print(f"****{channel_bins}")

                ########### Unit tests for the channel bins module ###########
                # print(f"Channel bins creation verification (should be all true): {iterx[0] == torch.tensor(channel_bins[0][-1])}")
                # print(f"Channel bins creation verification (should be all true): {iterx[14] == torch.tensor(channel_bins[14][-1])}")
                
                # # if iter_idx == 1:
                # #     exit()
                ##############################################################
            ################## Unit tests for one batch ##################
            # print(f"Check if labels and batch_lables are same : {labels,batch_labels}")
            # # Check if binning of channels is correct
            # print(f"channel_bins : length (num_channels) - {len(channel_bins)} | length of each element channel_bins[i] (batch_size) - {len(channel_bins[0])} | length of channel[i][j] (feature_size) : {len(channel_bins[0][0])}")
            # if batch_idx==2:
            #     exit()
            ##############################################################

        # Convert to numpy array
        channel_bins = np.array(channel_bins)
        labels = np.array(labels)

        ############################################# Saving the files #############################################
        # Generate the paths for saving the files
        saveDir = os.path.join(self.results_path, str(self.logcat_filter_rtime_threshold), str(truncated_duration), rnBin) 
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)

        print(f" - channel_bins stored at path : -\n {saveDir}")
        print(f" -> Shape of channel_bins (N_ch, N_samples, feature_size) and corresponding labels (N_samples) : {channel_bins.shape, labels.shape}\n ")
        
        # Paths where files needs to be saved
        channel_bins_path = os.path.join(saveDir, f'channel_bins_{split_type}.npy')
        labels_path = os.path.join(saveDir, f'labels_{split_type}.npy')
        filePath_path = os.path.join(saveDir, f'file_paths_{split_type}.npy')

        # Save the numpy array
        with open(channel_bins_path, 'wb') as f:
            np.save(f, channel_bins, allow_pickle=True)

        with open(labels_path, 'wb') as f:
            np.save(f, labels, allow_pickle=True)
        
        # Save the file paths
        with open(filePath_path, 'wb') as fp:
            pickle.dump(f_paths, fp)
        ############################################################################################################

        return [channel_bins_path, labels_path, filePath_path]


def main_worker(args, xmd_base_folder_location):
    """
    Worker node that performs the complete analysis.
    params:
        - args: easydict storing the experimental parameters
        - xmd_base_folder_location: Location of base folder of xmd
    """
    # Generate the feature engineered dataset for all logcat runtime thresholds and truncated durations for this dataset.
    feature_engineered_dataset.generate_feature_engineered_dataset(args, xmd_base_folder_location, 
                                                                   featEngineerDatasetFolderName="featureEngineeredDataset", 
                                                                   FolderName_featureEngineeredDatasetDetails="featureEngineeredDataset_details")

def main():
    ############################################## Setting up the experimental parameters ##############################################
    # Location of the base folder of xmd 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    base_folder_location = os.path.join(dir_path.replace("/src",""),"")

    # Get the arguments and the config file object
    cfg, args = get_args(xmd_base_folder=base_folder_location)

    # Create a runs directory for this run and push the config files for this run in the directory
    args.run_dir = os.path.join(base_folder_location, 'runs', args.timestamp)
    if args.create_runs_dir and os.path.isdir(args.run_dir) is False:
        os.mkdir(args.run_dir)
        shutil.copyfile(args.default_config_file, os.path.join(args.run_dir, args.default_config_file.split('/')[-1]))
        shutil.copyfile(args.update_config_file, os.path.join(args.run_dir, args.update_config_file.split('/')[-1]))

        # Write the updated config file in final_config.yaml
        cfg.export_config(os.path.join(args.run_dir, 'final_config.yaml'))
    ####################################################################################################################################

    # Start the analysis
    main_worker(args=args, xmd_base_folder_location= base_folder_location)

if __name__ == '__main__':
    main()