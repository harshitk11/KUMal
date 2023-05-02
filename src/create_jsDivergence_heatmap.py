"""
(1) Generate distribution of HPC values for benign and malware samples
"""
from create_raw_dataset import dataset_generator_downloader, dataset_split_generator, custom_collator, arm_telemetry_data, BENIGN_LABEL, MALWARE_LABEL
import argparse
import os
import torch
import tqdm
import matplotlib.pyplot as plt
import re
import numpy as np
import pickle
from scipy.stats import entropy
import seaborn as sns
import pandas as pd

class HPC_distribution_visualization():
    """
    Contains functions for generating the distribution of HPC values for benign and malware samples
    """
    def __init__(self, args):
        self.args = args
        self.rnList = [1,2,3,4] 
        self.num_channels = 3 # 3 channels for every rn
        
    def generate_HPC_distribution_benign_vs_malware(self, dataset_base_folder):
        """
        High-level pseudocode:
            - Create dataloader
            - Enumerate through all the samples and store the HPC values in a list
            - Plot the distribution of HPC values for benign and malware samples
            
        Output:
            - histograms_label_benign = {rn: {channel: [] for channel in range(self.num_channels)} for rn in self.rnList}
            - histograms_label_malware = {rn: {channel: [] for channel in range(self.num_channels)} for rn in self.rnList}
        """
        # Output
        histograms_label_benign = {rn: {channel: [] for channel in range(self.num_channels)} for rn in self.rnList}
        histograms_label_malware = {rn: {channel: [] for channel in range(self.num_channels)} for rn in self.rnList}          
        
        ds_split = dataset_split_generator(seed=10, partition_dist=[1,0])        
        HPC_partition_for_HPC_individual, HPC_partition_labels_for_HPC_individual = ds_split.create_HPC_partitions_and_labels_for_all_rn(base_location=dataset_base_folder)
        
        for rnval in range(4):
            # Samples for the corresponding rn
            dataset_c = arm_telemetry_data(partition=HPC_partition_for_HPC_individual[rnval], 
                                    labels=HPC_partition_labels_for_HPC_individual[rnval],
                                    split="train")
            
            custom_collate_fn = custom_collator(args=self.args)
            dataloader = torch.utils.data.DataLoader(
                    dataset_c,
                    num_workers=10,
                    batch_size=20,
                    collate_fn=custom_collate_fn,
                    shuffle=False)
            
            for batch_indx,(batch_list, batch_labels, batch_paths) in enumerate(tqdm.tqdm(dataloader)):                
                for iter_idx,iterx in enumerate(batch_list):
                    
                    # Extract the rn from the path
                    regex_pattern = r'.*\/(.*)__.*it(\d*)_rn(\d*).txt'
                    file_hash_obj = re.search(regex_pattern, batch_paths[iter_idx], re.M|re.I)
                    if file_hash_obj: 
                        rn_val = int(file_hash_obj.group(3).strip())
                        assert rn_val == rnval+1, "rn value is not equal to the expected value"
                    else:
                        raise ValueError(f"Regex pattern did not match for {batch_paths[iter_idx]}")   
                    
                    assert self.num_channels == iterx.shape[0], "Number of channels is not equal to 3"
                    
                    for channel in range(self.num_channels):
                        channel_values = iterx[channel].numpy()  # Convert tensor to numpy array
                        label = batch_labels[iter_idx].numpy()  # Convert tensor to numpy array

                        if label == BENIGN_LABEL:
                            histograms_label_benign[rn_val][channel].extend(channel_values)
                        elif label == MALWARE_LABEL:
                            histograms_label_malware[rn_val][channel].extend(channel_values)
                        else:
                            raise ValueError(f"Label value is not valid: {label}")
                
                
        return histograms_label_benign, histograms_label_malware
    
    @staticmethod
    def log_transform(data):
        return np.log1p(data)
    
    @staticmethod
    def remove_zeros(data):
        return [x for x in data if x != 0]
    
    @staticmethod
    def jensen_shannon_divergence(p, q):
        """
        Calculate the Jensen-Shannon Divergence between two probability distributions.
        Args:
            p (np.ndarray): First probability distribution.
            q (np.ndarray): Second probability distribution.
        Returns:
            float: Jensen-Shannon Divergence value.
        """
        p = np.asarray(p)
        q = np.asarray(q)
        m = (p + q) / 2
        return (entropy(p, m) + entropy(q, m)) / 2

    @staticmethod
    def plot_histograms(histograms_label_benign, histograms_label_malware, savepath=None):
        """
        Function to plot the histograms
        Output:
            - js_dict = {rn : {channel: js_divergence_score for channel in range(num_channels)} for rn in range(1, num_rn + 1)}

        """
        num_rn = len(histograms_label_benign)
        num_channels = len(histograms_label_benign[1])
        js_dict = {rn : {channel: None for channel in range(num_channels)} for rn in range(1, num_rn + 1)}

        for rn in range(1, num_rn + 1):
            for channel in range(num_channels):
                benign_data = histograms_label_benign[rn][channel]
                malware_data = histograms_label_malware[rn][channel]

                # Remove zeros and log transform the data
                benign_data = HPC_distribution_visualization.remove_zeros(benign_data)
                malware_data = HPC_distribution_visualization.remove_zeros(malware_data)
                benign_data = HPC_distribution_visualization.log_transform(benign_data)
                malware_data = HPC_distribution_visualization.log_transform(malware_data)
                
                # Find common support for both histograms
                min_value = min(min(benign_data), min(malware_data))
                max_value = max(max(benign_data), max(malware_data))
                bins = np.linspace(min_value, max_value, num=100)

                ########### Calculate the Jensen-Shannon divergence ###########
                # Create histograms and normalize them to obtain probability distributions
                benign_hist, _ = np.histogram(benign_data, bins=bins, density=True)
                malware_hist, _ = np.histogram(malware_data, bins=bins, density=True)
                js_divergence = HPC_distribution_visualization.jensen_shannon_divergence(benign_hist, malware_hist)
                js_dict[rn][channel] = js_divergence
                ###############################################################

                if savepath:
                    # Plot histograms
                    plt.figure()
                    plt.hist(benign_data, bins=bins, alpha=0.5, label="Benign")
                    plt.hist(malware_data, bins=bins, alpha=0.5, label="Malware")
                    plt.xlabel("HPC Values")
                    plt.ylabel("Frequency")
                    plt.legend(loc="upper right")
                    plt.title(f"rn: {rn} | channel: {channel} | JS Divergence: {js_divergence:.4f}")
                    plt.savefig(os.path.join(savepath, f"rn_{rn}_channel_{channel}.png"), dpi=300)
                    plt.close()
        
        return js_dict
    
class jsDivergence_td_lrt_gridGenerator:
    """
    Contains functions for generating the grid of LRT and TD values for the jensen shannon divergence
    """
    @staticmethod
    def LRT_vs_TD_grid_for_histogram(args, codebase_folder_location):
        """
        Generates histogram (benign,malware) for the grid containing LRT (logcat runtime threshold) and TD (truncated duration)
        Assumes that the dataset is already downloaded at the location specified by dataset_base_folder
        """
        # List of LRT and TD values
        candidate_LRT = [0, 10, 20, 30, 40, 50, 60, 70, 80]
        candidate_TD = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        
        # Local path list without any filter
        dataset_name = "std-dataset"
        dataset_base_folder=f"/hdd_6tb/hkumar64/arm-telemetry/kumal_dataset_tsne_visualization/0_0_0"
        dataset_generator_instance = dataset_generator_downloader(filter_values= [0,0,0], dataset_name=dataset_name, base_download_dir=dataset_base_folder)
        _,_,candidateLocalPathDictOrig = dataset_generator_instance.generate_dataset_winter(download_file_flag=False)

        for lrt in candidate_LRT:
            histogram_grid_TD_LRT = {lrt: {td: None for td in candidate_TD}}
            
            # Get the list of local paths
            filter_values = [lrt,0,0] #[runtime_per_file, num_logcat_lines_per_file, freq_logcat_event_per_file]
            dataset_generator_instance = dataset_generator_downloader(filter_values= filter_values, dataset_name=dataset_name, base_download_dir=dataset_base_folder)
            _,_,candidateLocalPathDictNew = dataset_generator_instance.generate_dataset_winter(download_file_flag=False)    
            num_benign, num_malware = dataset_generator_instance.count_number_of_apks() 
            print(f" - Number of benign apk: {num_benign} | Number of malware apk: {num_malware}")
        
            # Based on the list and the previous timestamplist, generate the list of files to be deleted from the downloaded dataset
            deleteFilePaths = {}
            for pathLabel,pathList in candidateLocalPathDictNew.items():
                deleteFilePaths[pathLabel] = [x for x in candidateLocalPathDictOrig[pathLabel] if x not in pathList]

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
            candidateLocalPathDictOrig = candidateLocalPathDictNew
                
            for td in candidate_TD:
                print(f" - LRT: {lrt} | TD: {td}")
                # Calculate the histogram for the given LRT and TD
                dataset_base_folder_full_location = os.path.join(dataset_base_folder, dataset_name)
                args.truncated_duration = td
                
                HPC_distribution_visualization_instance = HPC_distribution_visualization(args=args)
                histograms_label_benign, histograms_label_malware = HPC_distribution_visualization_instance.generate_HPC_distribution_benign_vs_malware(dataset_base_folder=dataset_base_folder_full_location)
                
                histogram_grid_TD_LRT[lrt][td] = (histograms_label_benign, histograms_label_malware)
        
            # Save the histogram grid as a json file (one json per lrt) 
            save_path_base_folder = os.path.join(codebase_folder_location, "res/hpc_histogram_pkl_files")
            with open(os.path.join(save_path_base_folder, f"histogram_grid_TD_LRT_{lrt}.pkl"), 'wb') as f:
                pickle.dump(histogram_grid_TD_LRT, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        
    @staticmethod
    def LRT_vs_TD_grid_for_jensen_shannon_divergence(codebase_folder_location):
        candidate_LRT = [0, 10, 20, 30, 40, 50, 60, 70, 80]
        js_grid_TD_LRT = {}
        for lrt in candidate_LRT:
            # Load the histogram grid
            save_path_base_folder = os.path.join(codebase_folder_location, "res/hpc_histogram_pkl_files")
            with open(os.path.join(save_path_base_folder, f"histogram_grid_TD_LRT_{lrt}.pkl"), 'rb') as f:
                histogram_grid_TD_LRT = pickle.load(f)
                
            js_grid_TD_LRT[lrt] = {}
            for td, histograms in histogram_grid_TD_LRT[lrt].items():        
                print(f" - LRT: {lrt} | TD: {td}")
                # save_path_plots = os.path.join(codebase_folder_location, "plots/dataset_characterization/HPC_distribution_plots")
                save_path_plots = None
                js_grid_TD_LRT[lrt][td] = HPC_distribution_visualization.plot_histograms(histograms[0], histograms[1], savepath=save_path_plots)
    
        # Save js_grid_TD_LRT as a pickle file
        with open(os.path.join(save_path_base_folder, f"js_grid_TD_LRT.pkl"), 'wb') as f:
            pickle.dump(js_grid_TD_LRT, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def visualize_js_grid(codebase_folder_location):
        # Read the js_grid_TD_LRT pickle file
        save_path_base_folder = os.path.join(codebase_folder_location, "res/hpc_histogram_pkl_files")
        with open(os.path.join(save_path_base_folder, f"js_grid_TD_LRT.pkl"), 'rb') as f:
            js_grid_TD_LRT = pickle.load(f)
        
        # Get the unique values for lrt, td, rn, and channel
        lrt_values = [0, 10, 20, 30, 40, 50, 60, 70, 80]
        td_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        rn_values = [1,2,3,4]
        channel_values = [0,1,2]

        # Create a dataframe for storing the mean values for each lrt and td combination
        mean_js_values = pd.DataFrame(index=td_values, columns=lrt_values)

        for lrt in lrt_values:
            for td in td_values:
                mean_js = np.mean([js_grid_TD_LRT[lrt][td][rn][channel]
                                for rn in rn_values
                                for channel in channel_values])
                mean_js_values.at[td, lrt] = mean_js

        mean_js_values = mean_js_values.astype(float)
        print("Heatmap of mean JS values")
        print(mean_js_values)
        
        # Create the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(mean_js_values, cmap="gray_r", annot=True, fmt=".4f", cbar=False, annot_kws={"size": 15, "weight": "bold"})
        # plt.title("Mean JS Values Heatmap", fontsize=14, fontweight="bold")
        plt.xlabel("Logcat-Runtime-Threshold", fontsize=16, fontweight="bold")
        plt.ylabel("Truncated-Duration", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.xticks(fontsize=14, fontweight="bold")
        plt.yticks(fontsize=14, fontweight="bold")
        plt.gca().invert_yaxis()
        savepath = os.path.join(codebase_folder_location, "plots/dataset_characterization/mean_js_values_heatmap.pdf")
        plt.savefig(savepath, dpi=300)

def download_dataset():
    """
    Downloads the dataset (without any filter) and returns the base folder location where the dataset is downloaded
    Output:
        - dataset_base_folder: Base folder location where the dataset is downloaded
    """
    # Parameters for downloading the dataset
    dataset_name = "std-dataset"
    filter_values = [0,0,0] #[runtime_per_file, num_logcat_lines_per_file, freq_logcat_event_per_file]
    string_filter_values = '_'.join(map(str, filter_values))
    dataset_base_folder=f"/hdd_6tb/hkumar64/arm-telemetry/kumal_dataset_tsne_visualization/{string_filter_values}"
    
    # Download the raw dataset if not already downloaded
    if not os.path.exists(dataset_base_folder):    
        dataset_generator_instance = dataset_generator_downloader(filter_values= filter_values, dataset_name=dataset_name, base_download_dir=dataset_base_folder)
        dataset_generator_instance.generate_dataset_winter(download_file_flag=True, num_download_threads=50)
        num_benign, num_malware = dataset_generator_instance.count_number_of_apks() 
        print(f" - Number of benign apk: {num_benign} | Number of malware apk: {num_malware}")
        
    return dataset_base_folder

def main():
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Arguments for creating the tsne visualizations')
    parser.add_argument('--truncated_duration', type=float, default=10, help='Duration for truncating HPC time series')
    parser.add_argument('--collected_duration', type=float, default=90.0, help='Duration for which data is collected')
    parser.add_argument('--num_histogram_bins', type=int, default=32, help='Number of histogram bins for feature engineering')
    parser.add_argument('--feature_engineering_flag', type=bool, default=False, help='Specify whether to perform feature engineering or not')
    # Parse the arguments
    args = parser.parse_args()
    
    # Location of the base folder where the codebase is located
    dir_path = os.path.dirname(os.path.realpath(__file__))
    codebase_folder_location = os.path.join(dir_path.replace("/src",""),"")

    # dataset_base_folder = download_dataset()
    # jsDivergence_td_lrt_gridGenerator.LRT_vs_TD_grid_for_histogram(args)
    # jsDivergence_td_lrt_gridGenerator.LRT_vs_TD_grid_for_jensen_shannon_divergence(codebase_folder_location)
    jsDivergence_td_lrt_gridGenerator.visualize_js_grid(codebase_folder_location)
    
    
    
    
if __name__ == '__main__':
    main()