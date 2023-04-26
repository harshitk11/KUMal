'''
Python script to create the dataset containing the raw GLOBL and HPC files.
'''

import torch
import contextlib
import time
import random
import os
from os import listdir, mkdir
from os.path import isfile, join, isdir
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import dropbox
import re
import numpy as np
import pickle
import traceback
from multiprocessing import Pool, Process
import argparse

BENIGN_LABEL = 0
MALWARE_LABEL = 1

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
    print(f'******* Local download location :{download_path} *******')
    while '//' in path:
        path = path.replace('//', '/')
    with stopwatch('download'):
        try:
            dbx.files_download_to_file(download_path, path)
        except (dropbox.exceptions.HttpError, dropbox.exceptions.ApiError) as err:
            print('[download-error]', err)
            return None


class arm_telemetry_data(torch.utils.data.Dataset):
    ''' 
    This is the Dataset object.
    A Dataset object loads the training or test data into memory.
    Your custom Dataset class MUST inherit torch's Dataset
    Your custom Dataset class should have 3 methods implemented (you can add more if you want but these 3 are essential):
    __init__(self) : Performs data loading
    __getitem__(self, index) :  Will allow for indexing into the dataset eg dataset[0]
    __len__(self) : len(dataset)
    '''

    def __init__(self, partition, labels, split):
        '''
            -labels = {file_path1 : 0, file_path2: 0, ...}

            -partition = {'train' : [file_path1, file_path2, ..],
                                'trainSG' : [file_path1, file_path2, ..],
                                'val' : [file_path1, file_path2]}

            -split = 'train', or 'test'                    
        '''
        if(split not in ['train','test']):
            raise NotImplementedError('Can only accept Train, Test')

        # Store the list of paths (ids) in the split
        self.path_ids = partition[split] 
        # Store the list of labels
        self.labels = labels
        
    def __len__(self):
        return len(self.path_ids)
     
    def __getitem__(self, idx):
        """
        params:
            - idx: index of the sample to be returned
        Output:
            - X: the hpc tensor of shape (Nchannels,T)
            - y: the corresponding label
            - id: the corresponding file path that contains the data
        """
        # Select the sample [id = file path of the dvfs file]
        id = self.path_ids[idx]

        # Get the label and hpc tensor
        y = self.labels[id]
        X = self.read_simpleperf_file(id)

        return X,y,id

    def read_simpleperf_file(self, f_path):
        '''
        Parses the simpleperf file at path = fpath, parses it and returns a tensor of shape (Nchannels,T)
        '''
        # Extract the rn value to select the perf channels
        rn_obj = re.search(r'.*_(rn\d*).txt', f_path, re.M|re.I)
        if rn_obj:
            rn_val = rn_obj.group(1).strip()

        # Dict storing the regex patterns for extracting the performance counter channels
        perf_channels = {
                    'rn1' : [r'(\d*),cpu-cycles',r'(\d*),instructions', r'(\d*),raw-bus-access'], 
                    'rn2' : [r'(\d*),branch-instructions',r'(\d*),branch-misses', r'(\d*),raw-mem-access'], 
                    'rn3' : [r'(\d*),cache-references',r'(\d*),cache-misses', r'(\d*),raw-crypto-spec'],
                    'rn4' : [r'(\d*),bus-cycles',r'(\d*),raw-mem-access-rd', r'(\d*),raw-mem-access-wr']
                }

        # Store the parsed performance counter data. Each item is one collection point constaining 3 performance counter [perf1,perf2,perf3]
        # perf_list = [[perf1_value1,perf2_value1,perf3_value1], [perf1_value2,perf2_value2,perf3_value2], [], ....]
        perf_list = [] 

        with open(f_path) as f:
            for line in f:
                ######################### Perform a regex search on this line #########################
                # Every new collection point starts with "Performance counter statistics,". We use this as a start marker.
                startObj = re.search(r'(Performance counter statistics)', line, re.M|re.I)
                if startObj: # A new collection point has started. Start an empty list for this collection point.
                    collection_point = []

                # Parse the first performance counter
                perf1Obj = re.search(perf_channels[rn_val][0], line, re.M|re.I)
                if perf1Obj: 
                    collection_point.append(float(perf1Obj.group(1).strip()))
                    
                # Parse the second performance counter
                perf2Obj = re.search(perf_channels[rn_val][1], line, re.M|re.I)
                if perf2Obj: 
                    collection_point.append(float(perf2Obj.group(1).strip()))
                
                # Parse the third performance counter
                perf3Obj = re.search(perf_channels[rn_val][2], line, re.M|re.I)
                if perf3Obj: 
                    collection_point.append(float(perf3Obj.group(1).strip()))

                # Every collection point ends with "Total test time" followed by the timestamp of collection
                endObj = re.search(r'Total test time,(.*),seconds', line, re.M|re.I)
                if endObj: # End of collection point reached
                    
                    collection_point = [float(endObj.group(1).strip())] + collection_point
                    
                    # Also perform a sanity check to make sure all the performance counters for the collection point are present.
                    if len(collection_point) != 4:
                        raise ValueError("Missing performance counter collection point")
                    else:
                        # We have all the data points. Add it to the list.
                        perf_list.append(collection_point)

        # Convert the list to a tensor (Shape : Num_data_points x Num_channels)
        perf_tensor = torch.tensor(perf_list, dtype=torch.float32)
        # Transpose the tensor to shape : Num_channels x Num_data_points
        perf_tensor_transposed = torch.transpose(perf_tensor, 0, 1)

        return perf_tensor_transposed
                
        

class custom_collator(object):
    def __init__(self, args):
        # Parameters for truncating the hpc time series. Consider the first truncated_duration seconds of the iteration
        self.truncated_duration = args.truncated_duration
        # Duration for which data is collected 
        self.cd = args.collected_duration 
        # Feature engineering parameters for simpleperf files
        self.num_histogram_bins = args.num_histogram_bins
        # Flag to indicate if we want to feature engineer
        self.reduced_feature_flag = args.feature_engineering_flag
        
    def __call__(self, batch):
        '''
        Takes a batch of files, outputs a tensor of of batch, the corresponding labels, and the corresponding file paths
        - If reduced_feature_flag is False, then will return a list instead of a stacked tensor, for both dvfs and simpleperf
        '''    
        # batch_hpc : [iter1, iter2, ... , iterB]  (NOTE: iter1 - Nchannels x T1 i.e. Every iteration has a different length. Duration of data collection is the same. Sampling frequency is different for each iteration)
        # batch_labels : [iter1_label, iter2_label, ...iterB_label]
        # batch_paths : [iter1_path, iter2_path, ...iterB_path]
        batch_hpc, batch_labels, batch_paths = list(zip(*batch))

        if self.reduced_feature_flag:
            # Stores the dimension reduced hpc for each batch
            reduced_batch_hpc = []

            # Divide the individual variates of the tensor into num_histogram_bins. And sum over the individual intervals to form feature size of 32 for each variate.
            for hpc_iter_tensor in batch_hpc:
                # Take the truncated duration of the tensor
                hpc_iter_tensor = self.truncate_hpc_tensor(hpc_iter_tensor)
                
                ## hpc_intervals : [chunks of size - Nchannels x chunk_size] where chunk_size = lengthOfTimeSeries/self.num_histogram_bins
                hpc_intervals = torch.tensor_split(hpc_iter_tensor, self.num_histogram_bins, dim=1)
                
                # Take sum along the time dimension for each chunk to get chunks of size -  Nchannels x 1
                sum_hpc_intervals = [torch.sum(hpc_int,dim=1, keepdim=False) for hpc_int in hpc_intervals]
                                    
                # Concatenate the bins to get the final feature tensor
                hpc_feature_tensor = torch.cat(sum_hpc_intervals, dim=0)
                
                # Adding one dimension for channel [for compatibility purpose]. N_Channel = 1 in this case.
                reduced_batch_hpc.append(torch.unsqueeze(hpc_feature_tensor, dim=0)) 
                
            batch_tensor = torch.stack(reduced_batch_hpc, dim=0)
        
        else:
            # NOTE: This is not a tensor. It is a list of the iterations.
            batch_tensor = batch_hpc 

        return batch_tensor, torch.tensor(batch_labels), batch_paths


    def truncate_hpc_tensor(self, hpc_tensor):
        """
        Truncates the hpc tensor (Nch x Num_datapoints) based on the value provided in self.truncated_duration

        params:
            - hpc_tensor: hpc tensor of shape Nch x Num_datapoints with 0th channel containing the time stamps

        Output:
            - truncated_hpc_tensor: truncated hpc tensor with the time channel removed
        """
        # Get the index of the timestamp in the hpc_tensor
        timestamp_array = np.round(hpc_tensor[0].numpy(), decimals=1)
        
        # If the truncated duration is more than the length of collection duration, then return the last collected time stamp
        if self.truncated_duration > np.amax(timestamp_array):
            truncation_index = len(timestamp_array)
        else:
            # truncation_index = np.where(timestamp_array == self.truncated_duration)[0][0]+1
            truncation_index = custom_collator.find_index_of_nearest(timestamp_array, self.truncated_duration)+1

        # Truncate the tensor using the index
        truncated_hpc_tensor = hpc_tensor[:,:truncation_index]
        # Remove the time axis
        truncated_hpc_tensor = truncated_hpc_tensor[1:]
        
        return truncated_hpc_tensor

    @staticmethod
    def find_index_of_nearest(array, value):
        """
        Returns the index of the element in the array which is closest to value.
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    

def get_dataloader(args, partition, labels, custom_collate_fn, required_partitions):
    '''
    Returns the dataloader objects for the different partitions.

    params: 
        -partition = {'train' : [file_path1, file_path2, ..],
                            'test' : [file_path1, file_path2, ..]}
                            
        -labels : {file_path1 : 0, file_path2: 1, ...}  (Benigns have label 0 and Malware have label 1)
        
        -custom_collate_fn : Custom collate function object (Used for feature reduction)

        -required_partitions : required_partitions = {"train":T or F, "test":T or F}           
   
    Output: 
        - trainloader, testloader : Dataloader object for train and test data.
    '''
    trainloader, testloader = None, None

    # Initialize the custom dataset class for training, validation, and test data
    if required_partitions["train"]:
        ds_train_full = arm_telemetry_data(partition, labels, split='train')
        trainloader = torch.utils.data.DataLoader(
            ds_train_full,
            num_workers=args.num_workers,
            batch_size=args.train_batchsz,
            collate_fn=custom_collate_fn,
            shuffle=args.train_shuffle,
        )
    
    if required_partitions["test"]:
        ds_test_full = arm_telemetry_data(partition, labels, split='test', file_type= file_type, normalize=normalize_flag)
        testloader = torch.utils.data.DataLoader(
            ds_test_full,
            num_workers=args.num_workers,
            batch_size=args.test_batchsz,
            collate_fn=custom_collate_fn,
            shuffle=args.test_shuffle,
            sampler = torch.utils.data.SequentialSampler(ds_test_full)
        )

    return trainloader, testloader


class dataset_split_generator:
    
    """
    Generates the dataset splits for the classification tasks.

    - Given a dataset, we have to handle the split [num_train_%, num_test_%] according to the following cases:
 
        1. If the dataset is used for training the models (i.e. std-dataset), then we create splits for training the classifier (num_train_% = 70%)
            and for testing the classifier (num_test_% = 30%). 
            
        2. If the dataset is used for testing the models (i.e., cd-year1-dataset etc.), then we use the entire dataset for testing the models (num_test_% = 100%) and there is no training split.
 
    """
    
    def __init__(self, seed, partition_dist) -> None:
        """
        params:
            - seed : Used for shuffling the file list before generating the splits
            - partition_dist = [num_train_%, num_test_%]
                                - num_train_% : percentage training samples
                                - num_test_% : percentage test samples
        """
        self.seed = seed
        self.partition_dist = partition_dist
        

    @staticmethod
    def create_labels_from_filepaths(benign_filepaths = None, malware_filepaths = None):
        '''
        Function to create a dict containing file location and its corresponding label
        Input : -benign_filepaths - List of file paths of the benign logs
                -malware_filepaths - List of file paths of the malware logs
        
        Output : -benign_label = {file_path1 : 0, file_path2: 0, ...}  (Benigns have label 0)
                -malware_label = {file_path1 : 1, file_path2: 1, ...}  (Malware have label 1)   
        '''

        # Create the labels dict from the list
        if benign_filepaths is not None:
            benign_label = {path: BENIGN_LABEL for path in benign_filepaths}
        
        if malware_filepaths is not None:
            malware_label = {path: MALWARE_LABEL for path in malware_filepaths} 

        if benign_filepaths is None: # Just return the malware labels
            return malware_label
        
        elif malware_filepaths is None: # Just return the benign labels
            return benign_label

        elif (benign_filepaths is None) and (malware_filepaths is None):
            raise ValueError('Need to pass arguments to create_labels_from_filepaths()')

        return benign_label, malware_label
    
    def create_splits(self, benign_label=None, malware_label=None):
        '''
        Function for splitting the dataset into Train and Test
        NOTE: If any of benign_label or malware_label is not passed as argument, then we ignore that, and
            create splits from whatever is passed as argument.

        Input : -benign_label = {file_path1 : 0, file_path2: 0, ...}  (Benigns have label 0)
                -malware_label = {file_path1 : 1, file_path2: 1, ...}  (Malware have label 1)
                -self.partition_dist = [num_train_%, num_test_%]

        Output : -partition = {'train' : [file_path1, file_path2, ..],
                                'test' : [file_path1, file_path2]}

        NOTE: 
            - partition may be empty for certain splits, e.g., when num_train_%=0 then 'train' partition is an empty list.
            - if num_train_% != 0, then the split between train and test will ensure that test contains hashes that are not present in train (to prevent contamination).
        '''
        # Fix the seed value of random number generator for reproducibility
        random.seed(self.seed) 
        
        # Create the partition dict (This is the output.)
        partition = {'train':[], 'test':[]}   

        ################################## Handling the benign labels ##################################
        if benign_label is not None:
            # Shuffle the dicts of benign and malware: Convert to list. Shuffle. 
            benign_label_list = list(benign_label.items())

            # Calculate the number of training, and test samples
            num_train_benign, num_test_benign = [math.ceil(x * len(benign_label)) for x in self.partition_dist]

            # Dividing the list of benign files into training, trainSG, and test buckets
            benign_train_list = benign_label_list[:num_train_benign]
            benign_test_list = benign_label_list[num_train_benign:num_train_benign+num_test_benign]
            random.shuffle(benign_train_list)
            
            # Add items in train list to train partition
            for path,label  in benign_train_list:
                partition['train'].append(path)

            # Add items in test list to test partition
            for path,label  in benign_test_list:
                partition['test'].append(path)
        ################################################################################################
        ################################## Handling the malware labels #################################
        if malware_label is not None:
            # Shuffle the dicts of benign and malware: Convert to list. Shuffle. 
            malware_label_list = list(malware_label.items())

            # Calculate the number of training, trainSG, and test samples
            num_train_malware, num_test_malware = [math.ceil(x * len(malware_label)) for x in self.partition_dist]

            # Dividing the list of malware files into training, trainSG, and test buckets
            malware_train_list = malware_label_list[:num_train_malware]
            malware_test_list = malware_label_list[num_train_malware:num_train_malware+num_test_malware]
            random.shuffle(malware_train_list)
            
            # Add items in train list to train partition
            for path,label  in malware_train_list:
                partition['train'].append(path)
                
            # Add items in test list to test partition
            for path,label  in malware_test_list:
                partition['test'].append(path)
        ################################################################################################
        # Shuffle the partitions
        random.shuffle(partition['train'])
        random.shuffle(partition['test'])

        return partition

    def create_all_datasets(self, base_location):
        """
        Function to create Train and Test splits for classification for the std and cd datasets.
        
        params: 
            - base_location : Location of the base folder. See the directory structure in create_matched_lists()
        Output:
            - Partition and partition labels for classification: (HPC_partition_for_HPC_individual,HPC_partition_labels_for_HPC_individual)
                                                                
        NOTE: Depending on the dataset type, certain partitions or labels will be empty. So you need to check for that in your code down the line.
        """
        print("********** Creating the splits [partitions and labels dict] for the classification tasks ********** ")
        
        ####### Creating a file list of malware and benign log files #######
        simpleperf_benign_rn_loc = [os.path.join(base_location, "benign","simpleperf",rn) for rn in ['rn1','rn2','rn3','rn4']]
        simpleperf_malware_rn_loc = [os.path.join(base_location, "malware","simpleperf",rn) for rn in ['rn1','rn2','rn3','rn4']]

        # Generate file_lists from these locations
        simpleperf_benign_file_list = [[join(_path,f) for f in listdir(_path) if isfile(join(_path,f))] for _path in simpleperf_benign_rn_loc]
        simpleperf_malware_file_list = [[join(_path,f) for f in listdir(_path) if isfile(join(_path,f))] for _path in simpleperf_malware_rn_loc]

        # Sort so that the files from same hash are grouped together
        simpleperf_benign_file_list = [sorted(_list) for _list in simpleperf_benign_file_list]
        simpleperf_malware_file_list = [sorted(_list) for _list in simpleperf_malware_file_list]
        ###################################################################################################################################################

        ###################### Generating the labels ######################
        # HPC_partition_labels_for_HPC_individual: [HPC_partition_labels_for_HPC_individual_with_rn1, HPC_partition_labels_for_HPC_individual_with_rn2, ...rn3, ...rn4]
        HPC_partition_labels_for_HPC_individual = []
        for indx in range(4):
            # You can use all the files for a given rn (not just the matched files) for creating labels.
            all_benign_label, all_malware_label = dataset_split_generator.create_labels_from_filepaths(benign_filepaths= simpleperf_benign_file_list[indx], malware_filepaths= simpleperf_malware_file_list[indx])
            all_labels = {**all_benign_label,**all_malware_label}
            HPC_partition_labels_for_HPC_individual.append(all_labels) # One labels dict for each rn
        ###################################################################  
        
        #########################****************************** Creating the splits for Individual HPC *********************************##############################
        HPC_partition_for_HPC_individual = []
        
        # For each rn
        for rn_val in range(4):
            # Get the file list for malware and benign
            file_list_b = simpleperf_benign_file_list[rn_val]
            file_list_m = simpleperf_malware_file_list[rn_val]

            # Create labels
            benign_label, malware_label = dataset_split_generator.create_labels_from_filepaths(benign_filepaths= file_list_b, malware_filepaths= file_list_m)

            # Create partition dict from the labels [70-30 split for std-dataset and 0-100 split for cd-dataset]
            partition = self.create_splits(benign_label= benign_label,malware_label= malware_label)

            # Append it to HPC individual
            HPC_partition_for_HPC_individual.append(partition)
        #########################*******************************************************************************************************##############################        
        
        ################################ Unit tests for testing the HPC individual partitions ################################        
        print(f"-> Stats for HPC-individual with train-test split: {self.partition_dist}.")
        try:
            for rn_indx, rn_partition_dict in enumerate(HPC_partition_for_HPC_individual):
                print(f" - numFiles in rn bin : {rn_indx+1}")
                print(f"partition\tnumFiles")  
                for key,value in rn_partition_dict.items():
                    try:
                        print(f"{key}\t{len(value)}")
                    except:
                        print(f"{key}\t{None}")
        except:
            print(None)
        #######################################################################################################################
        
        return HPC_partition_for_HPC_individual, HPC_partition_labels_for_HPC_individual


class dataset_generator_downloader:
    def __init__(self, filter_values, dataset_type, base_download_dir):
        """
        Dataset generator : Downloads the dataset from the dropbox.

        params:
            - filter_values : Filter values for the logcat files
                            Format : [runtime_per_file, num_logcat_lines_per_file, freq_logcat_event_per_file]
            - dataset_type : Type of dataset that you want to create
                            Can take one of the following values : ["std-dataset","cdyear1-dataset","cdyear2-dataset","cdyear3-dataset","bench-dataset"]
            
        """
        self.filter_values = filter_values

        # Root directory of xmd [Used for accessing the different logs]
        self.root_dir_path = os.path.dirname(os.path.realpath(__file__)).replace("/src","")

        # Base directory where all the files are downloaded
        self.base_download_dir = base_download_dir
        self.dataset_type = dataset_type
        ############################### Generating black list for malware apks for all the datasets #################################
        vt_malware_report_path = os.path.join(self.root_dir_path, "res", "virustotal", "hash_VT_report_all_malware_vt10.json")
        
        # If the black list already exists, then it will load the previous black list. To generate the new blacklist, delete
        # the not_malware_hashlist at "xmd/res/virustotal"
        self.std_dataset_malware_blklst = self.get_black_list_from_vt_report(vt_malware_report_path, vtThreshold=2)
        #############################################################################################################################

    def get_black_list_from_vt_report(self, vt_malware_report_path, vtThreshold):
        """
        Get the list of malware apks with less than vtThreshold vt-positives. We will not process the logs 
        from these apks as malware.

        params:
            - vt_malware_report_path : Path of the virustotal report of the malware
            - vtThreshold : Threshold of detections below which we discard the malware sample
         
        Output:
            - not_malware : List of hashes of apks with 0 or 1 vt positive
        """
        # Location where the not_malware list is stored
        not_malware_list_loc = os.path.join(self.root_dir_path,"res","virustotal",f"not_malware_hashlist_vtthreshold_{vtThreshold}.pkl")

        # Check if the not_malware_hashlist is already created. If yes, then return the previous list
        if os.path.isfile(not_malware_list_loc):
            with open(not_malware_list_loc, 'rb') as fp:
                not_malware = pickle.load(fp)
            return not_malware
        
        # Load the virustotal report
        with open(file=vt_malware_report_path) as f:
            report = json.load(f)

        # List storing the malware with 0 or 1 positive results
        not_malware = []

        # Parsing the virustotal report
        malware_details = {}
        for hash, hash_details in report.items():
            # Store the malware hash, positives, total, percentage positive
            malware_details[hash] = {'positives':hash_details['results']['positives'],
                                    'total':hash_details['results']['total'],
                                    'percentage_positive':round((float(hash_details['results']['positives'])/float(hash_details['results']['total']))*100,2),
                                    'associated_malware_families':[avengine_report['result'] for _,avengine_report in hash_details['results']['scans'].items() if avengine_report['result']]}
            
            # Identify the malware apks with less than vtThreshold vt_positives
            if int(hash_details['results']['positives']) < vtThreshold:
                print(f" - Adding {hash} to the not malware list.")
                not_malware.append(hash)

        # Save the not_malware list as a pickled file
        with open(not_malware_list_loc, 'wb') as fp:
            pickle.dump(not_malware, fp)

        print(f" --------- {len(not_malware)} apks added to the not_malware list --------- ")    
        return not_malware

    @staticmethod
    def extract_hash_from_filename(file_list):
        """
        Extract hashes from the shortlisted files [Used for counting the number of apks for the std-dataset and the cd-dataset].
        params:
            - file_list : List of files from which the hashes needs to be extracted
        Output:
            - hash_list : List of hashes that is extracted from the file list
        """
        # To store the list of hashes
        hash_list = []

        for fname in file_list:
            # Extract the hash from the filename
            hashObj = re.search(r'.*_(.*).apk.*', fname, re.M|re.I)
            hash_ = hashObj.group(1)

            if hash_ not in hash_list:
                hash_list.append(hash_)

        return hash_list

    def filter_shortlisted_files(self, file_list):
        """
        Filters out the blacklisted files from the shortlisted files. 
        [Used for filtering out the blacklisted files from the malware apks in the std-dataset].
        params:
            - file_list : List of file names on which the filter needs to be applied
        
        Output:
            - filtered_file_list : List of file names after the filter has been applied
        """
        # For tracking the number of files that are filtered out
        num_files_filtered_out = 0
        
        # Storing the file names post filter
        filtered_file_list = []

        for fname in file_list:
            # Extract the hash from the filename
            hashObj = re.search(r'.*_(.*).apk.*', fname, re.M|re.I)
            hash_ = hashObj.group(1)

            # If the hash is not in the blklst, then add it to the filtered list
            if hash_ not in  self.std_dataset_malware_blklst:
                filtered_file_list.append(fname)
            else:
                num_files_filtered_out += 1

        print(f"- Number of malware files that are filtered out: {num_files_filtered_out}")
        return filtered_file_list

    def create_shortlisted_files(self, parser_info_loc, apply_filter = True):
        '''
        Function to create a list of shortlisted files, based on logcat
        Input: 
            - parser_info_loc : Location of the parser_info file
            - apply_filter : If True, then applies the filter. Else, no filter (in the case of benchmark benign files)
            
        Output: 
            - shortlisted_files : List containing the dropbox location of the shortlisted files
            - logcat_attributes_list : List containing the corresponding logcat attributes of the shortlisted files
        '''
        # List of locations of the shortlisted files [Output of this method]
        shortlisted_files = []
        # List of corresponding logcat attributes for the shortlisted files
        logcat_attributes_list = []

        # Load the JSON containing the parsed logcat info for each iteration of data collection (You need to run codes/dropbox_module.py to generate the file)
        with open(parser_info_loc,"r") as fp:
            data=json.load(fp)

        # Extracting the threshold values
        if apply_filter:
            # If cd-dataset or std-dataset, then apply the logcat filter
            runtime_thr, num_logcat_event_thr, freq_logcat_event_thr = self.filter_values
        else: 
            # No need to filter the benchmark dataset since benchmarks run to completion always
            runtime_thr, num_logcat_event_thr, freq_logcat_event_thr = [0,0,0]

        for apk_folder,value in data.items():
            # apk_folder = Path of apk logcat folder (Contains the apk name)
            # value = [Number of logcat files, {logcat_file_1: [avg_freq, num_logcat_lines, time_diff]}, {logcat_file_2: [avg_freq, num_logcat_lines, time_diff]}, ...]

            for ind in range(value[0]): # Value[0] = number of logcat files for each apk. Each logcat file has its own dict.
                i = ind + 1 # For indexing into the corresponding dict in the list.
                
                for file_name,logcat_attributes in value[i].items():
                    # file_name = Name of the logcat file
                    # logcat_attributes = [avg_freq, num_logcat_lines, time_diff]

                    if((logcat_attributes[0] > freq_logcat_event_thr) and (logcat_attributes[1] > num_logcat_event_thr) and (logcat_attributes[2] > runtime_thr)):
                        # File satisfies all the threshold, add the full location of the file to the list
                        shortlisted_files.append(apk_folder+'/'+file_name) 
                        logcat_attributes_list.append([logcat_attributes[0],logcat_attributes[1],logcat_attributes[2]])

        return shortlisted_files, logcat_attributes_list

    ######################################## Helper methods to download the files from dropbox ########################################
    @staticmethod
    def create_dropbox_location(shortlisted_files, file_type):
        '''
        Function to create a list of dropbox locations and corresponding locations on the local machine
        from the shortlisted files based on the file_type (dvfs, logcat, simpleperf)
        Input :
                - shortlisted_files : Full dropbox locations of the logcat files of the shortlisted files
                - file_type : (dvfs, logcat, simpleperf)
                
        Output : 
                -shortlisted_files_mod (List of dropbox locations for the given file_type)
                -localhost_loc (List of corresponding file locations on the local host)
        '''

        shortlisted_files_mod = [] # Contains the location in dropbox
        localhost_loc = [] # Contains the location of the file in the local host

        for location in shortlisted_files:

            # Extract the iter, rn, and base_locations
            inputObj = re.search(r'(\/.*\/)logcat\/(.*logcat)(.*)', location, re.M|re.I)
            base_loc = inputObj.group(1)
            file_loc = inputObj.group(2)
            iter_rn = inputObj.group(3)
            
            # Extract the rn number [Will be used for separating the HPC data into buckets]
            rn_obj = re.search(r'.*\_(.*)\.txt', iter_rn, re.M|re.I)
            rn_num = rn_obj.group(1) # Will be one of the following : ['rn1','rn2','rn3','rn4']
            
            # Extract the apk hash [Will inject the hash into the file name to accurately track the apks the logs are collected from]
            hash_obj = re.search(r'.*_(.*)\.apk', base_loc, re.M|re.I)
            apk_hash = hash_obj.group(1)
            
            if file_type == 'dvfs':
                new_loc = base_loc+'dvfs/'+file_loc.replace('logcat','devfreq_data')+iter_rn # Dropbox location
                rem_loc = 'dvfs/'+apk_hash+'_'+file_loc.replace('logcat','devfreq_data')+iter_rn # Location on the local host
            elif file_type == 'logcat':
                new_loc = location
                rem_loc = 'logcat/'+apk_hash+'_'+file_loc+iter_rn
            
            # For performanc counter, we have 4 buckets : rn1, rn2, rn3, rn4. 
            elif file_type == 'simpleperf':
                new_loc = base_loc+'simpleperf/'+file_loc.replace('_logcat','')+iter_rn.replace('iter_','it')
                
                # Create local location depending on the rn bucket
                if (rn_num == 'rn1'):
                    rem_loc = 'simpleperf/rn1/'+apk_hash+'_'+file_loc.replace('_logcat','')+iter_rn.replace('iter_','it')
                elif (rn_num == 'rn2'):
                    rem_loc = 'simpleperf/rn2/'+apk_hash+'_'+file_loc.replace('_logcat','')+iter_rn.replace('iter_','it')
                elif (rn_num == 'rn3'):
                    rem_loc = 'simpleperf/rn3/'+apk_hash+'_'+file_loc.replace('_logcat','')+iter_rn.replace('iter_','it')
                elif (rn_num == 'rn4'):
                    rem_loc = 'simpleperf/rn4/'+apk_hash+'_'+file_loc.replace('_logcat','')+iter_rn.replace('iter_','it')
                else :
                    ## Corner case due to parser failing. Move to the next location.
                    # print(rn_num, location)
                    # raise ValueError('Parser returned an incorrect run number')   
                    continue  
            
            else: 
                ## Corner case due to parser failing. Move to the next location.
                # print(file_type, location)
                # raise ValueError('Incorrect file type provided')
                # print(rn_num, location)
                # raise ValueError('Parser returned an incorrect run number')   
                continue

            shortlisted_files_mod.append(new_loc)
            localhost_loc.append(rem_loc)

        return shortlisted_files_mod, localhost_loc
    
    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
            
    def download_shortlisted_files(self, shortlisted_files, file_type, app_type, num_download_threads, download_flag):
        '''
        Function to download the shortlisted files from dropbox. 
        Download process starts only if download_flag is true, else a list of candidate local locations of files is returned from the shortlisted files.

        params : 
            -shortlisted_files : List containing the dropbox location of the shortlisted files
            -file_type : the file type that you want to download : 'logcat', 'dvfs', or, 'simpleperf'
            -app_type : 'malware' or 'benign'
            -num_download_threads : Number of simultaneous download threads.
            -download_flag : Download process starts if the flag is set to True
            
        Output : Downloads the shortlisted files in <root_dir>/data/<dataset_type>. 
                localhost_loc: Returns the list of local locations of downloaded files
                       
        '''
        # Create the download location on the local host
        base_download_location = os.path.join(self.base_download_dir, self.dataset_type, app_type)
        
        # Get the dropbox api key
        with open(os.path.join(self.root_dir_path,"src","dropbox_api_key")) as f:
            access_token = f.readlines()[0]

        # Authenticate with Dropbox
        print('Authenticating with Dropbox...')
        dbx = dropbox.Dropbox(access_token)
        print('...authenticated with Dropbox owned by ' + dbx.users_get_current_account().name.display_name)

        # Create the dropbox location for the give file_type from the shortlisted_files
        dropbox_location, localhost_loc = dataset_generator_downloader.create_dropbox_location(shortlisted_files, file_type)

        # Full localhost locations [this is the list of local locations which is returned by this function]
        full_localhost_loc = [os.path.join(base_download_location,lloc) for lloc in localhost_loc]

        # Counter to see how many files were not downloaded
        not_download_count = 0

        if download_flag:
            # Create folder locations. If file_type is simpleperf then create rn bucket folders for each of them.
            os.system(f'mkdir -p {os.path.join(base_download_location, file_type)}')
            if (file_type == 'simpleperf'):
                os.system(f"mkdir -p {os.path.join(base_download_location, file_type, 'rn1')}")
                os.system(f"mkdir -p {os.path.join(base_download_location, file_type, 'rn2')}")
                os.system(f"mkdir -p {os.path.join(base_download_location, file_type, 'rn3')}")
                os.system(f"mkdir -p {os.path.join(base_download_location, file_type, 'rn4')}")

            print("--------- Downloading all the shortlisted files ---------")
            if num_download_threads > 1:
                # Start the download [Downloads in parallel]
                for dbx_loc_chunk, local_loc_chunk in zip(dataset_generator_downloader.chunks(dropbox_location,num_download_threads),dataset_generator_downloader.chunks(localhost_loc,num_download_threads)):
                    arguments = ((dbx,dbx_loc,os.path.join(base_download_location,local_loc)) for dbx_loc,local_loc in zip(dbx_loc_chunk,local_loc_chunk))
                    processList = []              
                    try:
                        for arg in arguments:
                            download_process = Process(target=download, name="Downloader", args=arg)
                            processList.append(download_process)
                            download_process.start()
                        
                        for p in processList:
                            p.join()
                    except:
                        continue
                
            else:
                # Start the download [Downloads serially]
                for i, location in enumerate(dropbox_location):
                    try:
                        download(dbx, location, os.path.join(base_download_location, localhost_loc[i]))
                    except:
                        not_download_count+=1
                        traceback.print_exc()
                        print(f'File not downloaded : Count = {not_download_count}')

                # Print the total files not downloaded
                print(f" ******************* Total files not downloaded : {not_download_count} *******************")
        
        return full_localhost_loc

    ###################################################################################################################################
    def count_number_of_apks(self):
        """
        Count the number of apks (hashes) in the benign and malware file_list.
        params:
            - file_list: List of file names (including location)

        Output: 
            - num_apk_benign, num_apk_malware : Number of benign and malware apks
        """

        shortlisted_files_benign,shortlisted_files_malware, _ = self.generate_dataset_winter(download_file_flag=False)

        # Get the hash_list for benign and malware
        hashlist_benign = dataset_generator_downloader.extract_hash_from_filename(shortlisted_files_benign)
        hashlist_malware = dataset_generator_downloader.extract_hash_from_filename(shortlisted_files_malware)

        return len(hashlist_benign), len(hashlist_malware)


    def generate_dataset_winter(self, download_file_flag, num_download_threads=0):
        """
        Generates the dataset (benign,malware) based on the dataset_type and filter_values
        params:
            - download_file_flag : If True, then will download all the shortlisted files
            - num_download_threads : Number of simultaneous download threads. Only needed when download_file_flag is True.
            
        Output:
            - Generated dataset at the specified location
            - shortlisted_files_benign, shortlisted_files_malware (Corresponding dvfs and simpleperf files will be downloaded
                if download_file_flag is True.)
            - candidateLocalPathDict : Local locations of the files that should be downloaded
        """
        # 1. Create shortlisted files based on the logcat filter and dataset type
        if self.dataset_type == "std-dataset":
            # Get the location of the parser info files
            parser_info_benign1 = os.path.join(self.root_dir_path, "res/parser_info_files/kumal", f"parser_info_std_benign.json")
            parser_info_benign2 = os.path.join(self.root_dir_path, "res/parser_info_files/kumal", f"parser_info_std_benign_dev2.json")
            parser_info_malware1 = os.path.join(self.root_dir_path, "res/parser_info_files/kumal", f"parser_info_std_malware.json")
            parser_info_malware2 = os.path.join(self.root_dir_path, "res/parser_info_files/kumal", f"parser_info_std_vt10_malware_dev2.json")

            # Create shortlisted files for benign and malware
            shortlisted_files_benign1, _ = self.create_shortlisted_files(parser_info_loc=parser_info_benign1, apply_filter=True)
            shortlisted_files_benign2, _ = self.create_shortlisted_files(parser_info_loc=parser_info_benign2, apply_filter=True)
            shortlisted_files_malware1, _ = self.create_shortlisted_files(parser_info_loc=parser_info_malware1, apply_filter=True)
            shortlisted_files_malware2, _ = self.create_shortlisted_files(parser_info_loc=parser_info_malware2, apply_filter=True)

            shortlisted_files_benign = shortlisted_files_benign1+shortlisted_files_benign2
            shortlisted_files_malware = shortlisted_files_malware1+shortlisted_files_malware2

            # Filter out the blacklisted files from the malware file list
            shortlisted_files_malware = self.filter_shortlisted_files(shortlisted_files_malware)

            
        elif self.dataset_type == "cdyear1-dataset":
            # Get the location of the parser info files
            parser_info_benign = os.path.join(self.root_dir_path, "res/parser_info_files/kumal", f"parser_info_cd_year1_benign.json")
            parser_info_malware = os.path.join(self.root_dir_path, "res/parser_info_files/kumal", f"parser_info_cd_year1_malware.json")

            # Create shortlisted files for benign and malware
            shortlisted_files_benign, _ = self.create_shortlisted_files(parser_info_loc=parser_info_benign, apply_filter=True)
            shortlisted_files_malware, _ = self.create_shortlisted_files(parser_info_loc=parser_info_malware, apply_filter=True)

            # Filter out the blacklisted files from the malware file list
            shortlisted_files_malware = self.filter_shortlisted_files(shortlisted_files_malware)
        
        elif self.dataset_type == "cdyear2-dataset":
            # Get the location of the parser info files
            parser_info_benign = os.path.join(self.root_dir_path, "res/parser_info_files/kumal", f"parser_info_cd_year2_benign.json")
            parser_info_malware = os.path.join(self.root_dir_path, "res/parser_info_files/kumal", f"parser_info_cd_year2_malware.json")

            # Create shortlisted files for benign and malware
            shortlisted_files_benign, _ = self.create_shortlisted_files(parser_info_loc=parser_info_benign, apply_filter=True)
            shortlisted_files_malware, _ = self.create_shortlisted_files(parser_info_loc=parser_info_malware, apply_filter=True)

            # Filter out the blacklisted files from the malware file list
            shortlisted_files_malware = self.filter_shortlisted_files(shortlisted_files_malware)
        
        elif self.dataset_type == "cdyear3-dataset":
            # Get the location of the parser info files
            parser_info_benign = os.path.join(self.root_dir_path, "res/parser_info_files/kumal", f"parser_info_cd_year3_benign.json")
            parser_info_malware = os.path.join(self.root_dir_path, "res/parser_info_files/kumal", f"parser_info_cd_year3_malware.json")

            # Create shortlisted files for benign and malware
            shortlisted_files_benign, _ = self.create_shortlisted_files(parser_info_loc=parser_info_benign, apply_filter=True)
            shortlisted_files_malware, _ = self.create_shortlisted_files(parser_info_loc=parser_info_malware, apply_filter=True)

            # Filter out the blacklisted files from the malware file list
            shortlisted_files_malware = self.filter_shortlisted_files(shortlisted_files_malware)
        
            
        else:
            raise(ValueError("Incorrect dataset type specified"))

        #################### Dataset Info ####################
        print(f"Information for the dataset : {self.dataset_type}")
        print(f"- Number of benign files : {len(shortlisted_files_benign)}")
        print(f"- Number of malware files : {len(shortlisted_files_malware)}")
        ###################################################### 
        
        
        # 2. Download the shortlisted files at <root_dir>/data/<dataset_type> 
        
        # Downloading the shortlisted performance counter files [Needs to be executed only once to download the files]
        malware_simpeperf_path =  self.download_shortlisted_files(shortlisted_files_malware, file_type= 'simpleperf', app_type= 'malware', num_download_threads=num_download_threads, download_flag=download_file_flag)
        benign_simpleperf_path =  self.download_shortlisted_files(shortlisted_files_benign, file_type= 'simpleperf', app_type= 'benign', num_download_threads=num_download_threads, download_flag=download_file_flag)

        candidateLocalPathDict = {"malware_simpeperf_path": malware_simpeperf_path,
                                "benign_simpleperf_path": benign_simpleperf_path}

        return shortlisted_files_benign,shortlisted_files_malware, candidateLocalPathDict

def generate_apk_list_for_software_AV_comparison(dataset_name, saveHashList_flag):
    """
    Generates a list of apks for each dataset. This list is used to generate the apk database on which SOTA software based AV
    params:
        - dataset_name (str): Type of the dataset for which the apk list will be generated
        - saveHashList_flag (bool) : Flag to indicate if hashlist will be saved
    Output:
        - writes the hashlist_benign and hashlist_malware for the specified dataset in res/softwareAVcomparisonApkList
    """
    print(dataset_name)
    # dataset_generator_instance = dataset_generator_downloader(filter_values= [0,50,2], dataset_type=dataset_name, base_download_dir="/hdd_6tb/hkumar64/arm-telemetry/usenix_winter_dataset")
    dataset_generator_instance = dataset_generator_downloader(filter_values= [0,0,0], dataset_type=dataset_name, base_download_dir="/hdd_6tb/hkumar64/arm-telemetry/usenix_winter_dataset")
    shortlisted_files_benign,shortlisted_files_malware, _ = dataset_generator_instance.generate_dataset_winter(download_file_flag=False)
    hashlist_benign = dataset_generator_downloader.extract_hash_from_filename(shortlisted_files_benign)
    hashlist_malware = dataset_generator_downloader.extract_hash_from_filename(shortlisted_files_malware)
    num_benign, num_malware = dataset_generator_instance.count_number_of_apks() 
    print(f" - Number of benign apk: {num_benign, len(hashlist_benign)} | Number of malware apk: {num_malware, len(hashlist_malware)}")
   
    if saveHashList_flag:
        root_dir_path = os.path.dirname(os.path.realpath(__file__)).replace("/src","")
        baseLoc = os.path.join(root_dir_path, "res", "APK_hashList_final_dataset_no_filter")
        if not os.path.isdir(baseLoc):
            os.mkdir(baseLoc)
            
        benignHashList_location = os.path.join(baseLoc, f"{dataset_name}_benignHashList.pkl")
        malwareHashList_location = os.path.join(baseLoc, f"{dataset_name}_malwareHashList.pkl")

        with open(benignHashList_location, 'wb') as fp:
            pickle.dump(hashlist_benign, fp)
        with open(malwareHashList_location, 'wb') as fp:
            pickle.dump(hashlist_malware, fp)


def main():
    baseDownloadDir = "/hdd_6tb/hkumar64/arm-telemetry/kumal_dataset"
    # # STD-Dataset
    dataset_generator_instance = dataset_generator_downloader(filter_values= [0,50,2], dataset_type="std-dataset", base_download_dir=baseDownloadDir)
    # # CD-Dataset
    # dataset_generator_instance = dataset_generator_downloader(filter_values= [0,0,0], dataset_type="std-dataset", base_download_dir=baseDownloadDir)
    # # Bench-Dataset
    # dataset_generator_instance = dataset_generator_downloader(filter_values= [15,50,2], dataset_type="bench-dataset", base_download_dir=baseDownloadDir)    
    
    # shortlisted_files_benign,shortlisted_files_malware, candidateLocalPathDict = dataset_generator_instance.generate_dataset_winter(download_file_flag=False, num_download_threads=30)
    # num_benign, num_malware = dataset_generator_instance.count_number_of_apks() 
    # print(f" - Number of benign apk: {num_benign} | Number of malware apk: {num_malware}")
    # exit()


    # # ######################### Testing the datasplit generator #########################
    test_path = "/hdd_6tb/hkumar64/arm-telemetry/kumal_dataset/std-dataset"          
    x = dataset_split_generator(seed=10, partition_dist=[0.7,0.3])        
    HPC_partition_for_HPC_individual, HPC_partition_labels_for_HPC_individual = x.create_all_datasets(base_location=test_path)
    # exit()
    # # ###################################################################################
    
    # # ################# Testing the dataset class and dataloader ########################
    rnval = 0
    dataset_c = arm_telemetry_data(partition=HPC_partition_for_HPC_individual[rnval], 
                                   labels=HPC_partition_labels_for_HPC_individual[rnval],
                                   split="train")
    
    # # Print the first 10 items in the dataset (testing the dataset class)
    # for item_indx in range(10):
    #     X,y,id = dataset_c.__getitem__(item_indx)
    #     print(f"X.shape: {X.shape} | y: {y} | id: {id}")
    
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Example arguments')
    parser.add_argument('--truncated_duration', type=float, default=30,
                        help='Duration for truncating HPC time series')
    parser.add_argument('--collected_duration', type=float, default=90.0,
                        help='Duration for which data is collected')
    parser.add_argument('--num_histogram_bins', type=int, default=32,
                        help='Number of histogram bins for feature engineering')
    parser.add_argument('--feature_engineering_flag', type=bool, default=True,
                        help='Specify whether to perform feature engineering or not')
    # Parse the arguments
    args = parser.parse_args()
    custom_collate_fn = custom_collator(args=args)
    
    trainloader = torch.utils.data.DataLoader(
            dataset_c,
            num_workers=0,
            batch_size=10,
            collate_fn=custom_collate_fn,
            shuffle=False)
    
    trainIter = iter(trainloader)
    batch_tensor, batch_labels, batch_paths = next(trainIter)
    print(f"batch_tensor.shape: {batch_tensor.shape} | batch_labels: {batch_labels} | batch_paths: {batch_paths}")
    
    Nch, _, length = batch_tensor.shape

    fig, axs = plt.subplots(Nch, 1, figsize=(10, 5*Nch))

    for i in range(Nch):
        axs[i].plot(batch_tensor[i, 0, :])
        axs[i].set_title('Channel {}'.format(i))

    plt.savefig('test2.png')

    # # ###################################################################################
    
    
    # ######################### Generating hash list for software AV comparison #########################
    # saveHashList_flag = True
    # generate_apk_list_for_software_AV_comparison(dataset_name = "std-dataset", saveHashList_flag=saveHashList_flag)
    # generate_apk_list_for_software_AV_comparison(dataset_name = "cdyear1-dataset", saveHashList_flag=saveHashList_flag)
    # generate_apk_list_for_software_AV_comparison(dataset_name = "cdyear2-dataset", saveHashList_flag=saveHashList_flag)
    # generate_apk_list_for_software_AV_comparison(dataset_name = "cdyear3-dataset", saveHashList_flag=saveHashList_flag)
    # ###################################################################################################
    
    
    
if __name__ == '__main__':
    main()