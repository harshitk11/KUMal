"""
Contains all the utility classes and functions.
"""

import argparse
from genericpath import isfile
import json
import yaml
from easydict import EasyDict as edict
import re
from datetime import datetime
from datetime import timedelta
import statistics
from virus_total_apis import PublicApi as VirusTotalPublicApi
import time
import os
import pickle


class Config:
    """
    Python class to handle the config files.
    """
    def __init__(self, file) -> None:
        """
        Reads the base configuration file.
        
        params:
            - file: Path of the base configuration file.
        """
        # Reads the base configuration file
        self.args = self.read(file)
        print(f"---------------------------------------------------------------------------------------------------------------")
        print(f"Base configuration file : {file}")
        for key,val in self.args.items():
            print(f" - {key}: {val}")
        print(f"---------------------------------------------------------------------------------------------------------------")
    
    def update(self, updatefile):
        """
        This method updates the base-configuration file based on the values read from the updatefile.

        params:
            - updatefile: Path of the file containing the updated configuration parameters.
        """
        uArgs = self.read(updatefile)
        print(f"Updating the configuration using the file : {updatefile}")
        for key, val in uArgs.items():
            self.args[key] = val
            print(f" - {key} : {val}")

        print("Configuration file updated")
        print(f"---------------------------------------------------------------------------------------------------------------")


    @staticmethod
    def read(filename):
        """
        Reads the configuration file.
        """
        with open(filename, 'r') as f:
            parser = edict(yaml.load(f, Loader=yaml.FullLoader))
        return parser

    @staticmethod
    def print_config(parser):
        """
        Prints the args.
        params:
            - parser: edict object of the config file
        """
        print("========== Configuration File ==========")
        for key in parser:
            print(f" - {key}: {parser[key]}")

    def get_config(self):
        """
        Returns the currently stored config in the object.
        """
        return self.args

    def export_config(self, filename):
        """
        Writes the arguments in the file specified by filename (includes path)
        """
        with open(filename, 'w') as f:
            yaml.dump(dict(self.args), f)


class logcat_parser:
    """
    Class containing all the methods that aid in parsing the logcat file.
    
    We parse the logcat file to extract the following information from it:
        1) Number of lines 
        2) Timestamp difference to see how long the application executes
        3) Rate at which logcat events are happening
    """
    @staticmethod
    def extract_events(logcat_filename):
        """
        Function to read a logcat file and extract the list of timestamps
        
        params:
            - logcat_filename: specify with file path

        Output:
            - timestamp_list: list of timestamp of the events
        """
        # List to store the extracted time stamps
        timestamp_list = []
        rfile = open(logcat_filename, 'r', errors='ignore') # Ignore the encoding errors. Ref: 'https://docs.python.org/3/library/functions.html#open'

        # Extract the time stamp from each of the logcat event and store it in a list
        while True:
            try: 
                line = rfile.readline()
                
            except Exception as e: # Ignore the lines that throw errors [encoding issues with chinese characters]
                print(e)
                print("*************** Ignoring error ***************")
                continue
            
            else:    
                # Regex to extract the time stamp
                logcat_obj = re.match( r'\d\d-\d\d (\d\d:\d\d:\d\d\.\d\d\d)', line, re.M|re.I)

                if(logcat_obj):
                    # Add the timestamp to a list
                    timestamp_list.append(logcat_obj.group(1))
                    
                if not line: # If line is empty, you have reached the end of the file. (readline() returns an empty string when it reaches end of file)
                    break

        rfile.close()
        # Return list of timestamp of the events
        return timestamp_list
    
    @staticmethod
    def get_time_difference(tstamp_list):
        """
        Function to read the timestamp list and get the timestamp difference between the last and the first event
        params:
            - tstamp_list: List of timestamps
        Output:
            - timestamp difference
        """
        if (len(tstamp_list) > 0): # Need to have atleast one event in logcat to get the time difference
            start_time = tstamp_list[0]
            end_time = tstamp_list[-1]
            time_format = '%H:%M:%S.%f'
            
            t_delta = datetime.strptime(end_time, time_format) - datetime.strptime(start_time, time_format)

            # Corner case where interval might cross midnight
            if t_delta.days < 0:
                t_delta = timedelta(
                    days = 0,
                    seconds= t_delta.seconds,
                    microseconds= t_delta.microseconds
                )

            return t_delta.total_seconds()
        
        else:
            return 0

    @staticmethod
    def get_logcat_lines(tstamp_list):
        """ 
        Function to return the number of lines in the timestamp list
        params: 
            - tstamp_list: List of timestamps
        """
        return len(tstamp_list)

    @staticmethod
    def get_average_frequency(tstamp_list):
        """
        Function to calculate the average frequency of events using the timestamp list
        params: 
            - tstamp_list: List of timestamps
        """
        try:
            # You need to have atleast 2 timestamps in the list to get the time difference
            if (len(tstamp_list) > 1): 
                time_format = '%H:%M:%S.%f'
                
                # Calculate the time difference between successive events and store the difference in a list
                time_dif_list = [(datetime.strptime(tstamp_list[i], time_format)-datetime.strptime(tstamp_list[i-1], time_format)).total_seconds() for i in range(1, len(tstamp_list))]

                # Time difference between successive events can be negative sometimes [logcat issue], so we take mod before averaging
                time_dif_list = [abs(dif) for dif in time_dif_list]

                # Get the mean of the time difference 
                mean_time_dif = statistics.mean(time_dif_list)

                # Inverse of the time difference gives average frequency
                avg_freq = 1/mean_time_dif

            else: 
                avg_freq = 0
        
        except:
            # Error happens when mean of the time difference is 0
            avg_freq=0
            
        return avg_freq

class malware_label_generator:
    """
    Contains helper functions to generate VT reports for AVClass [https://github.com/malicialab/avclass] (for malware label generation)
    """
    @staticmethod
    def generate_hashlist(metainfo_path):
        """
        Generates the list of hashes from the metainfo file
        params:
            - metainfo_path: Path of the metainfo file from which the hash list needs to be extracted
        Output:
            - hashList: List of hashes extracted from the metainfo file
        """
        hashList = []
        
        with open(metainfo_path,'rb') as f:
            mInfo = json.load(f)

        for apkHash in mInfo:
            hashList.append(apkHash)

        return hashList

    @staticmethod
    def get_vt_report(hashList, outputFilePath):
        """
        Takes as input list of hashes and outputs a dict with key = hash and value = report
        
        params:
            - hashList: List of hashes for which the vt report needs to be generated
            - outputFilePath: Path of the output file where the report_dict will be dumped

        Output:
            - report_dict: key = hash and value = report
        """
        # Dict containing the hash: report pair [This is the final output]
        report_dict = {}
        # Checkpointing: If the report file already exists, then read it into report_dict 
        if os.path.isfile(outputFilePath):
            with open(outputFilePath,'rb') as fp:
                report_dict = json.load(fp)

        # VT api key
        API_KEY = 'bc49bca0c45ee170c3353cc5aefcabdb2decdb3d5f37dbd6ea32ea9fa9275b78'

        #Instantiate the VT API   
        vt = VirusTotalPublicApi(API_KEY)

        # MAX_REQUEST in a day-delay between the sample is adjusted accordingly
        MAX_REQUEST=480

        for indx, hash in enumerate(hashList):
            # Checkpointing: If the report already exists in report_dict then skip this hash
            if hash in report_dict:
                continue

            response = vt.get_file_report(hash)         
            if(response['response_code']==200 and response['results']['response_code']==1):
                report_dict[hash] = response
                with open(outputFilePath,'w') as fp:
                    json.dump(report_dict,fp, indent=2)
                positive=response['results']['positives']
                total=response['results']['total']
                print (f"- [{indx}] Hash : {hash} | Num positives : {positive} | Total : {total}")                
            else:
                print(response)
                print(f"- [{indx}] Skipping this app BAD Request or Not available in the repo : {hash}")

            # We want MAX_REQUEST requests in 1 day    
            time.sleep(int(24*60*60.0/MAX_REQUEST))

        return report_dict

    @staticmethod
    def generate_vt_report_all_malware(metaInfoPath, outputReportPath):
        """
        Generates VT report for all the malware in the all the datasets: STD, CDyear1, CDyear2, CDyear3
        
        params:
            - metaInfoPath: Base directory of the folder where all the meta info files are stored
            - outputReportPath: Path where the vt report file will be stored
        """
        dataset_type = ["std_vt10","cd_year1","cd_year2","cd_year3","std"]

        # Generating a combined hash list containing hashes of malware in all the datasets
        hashListAllMalware = []
        for datType in dataset_type:
            mPath = os.path.join(metaInfoPath, f"meta_info_{datType}_malware.json")
            hashListAllMalware += malware_label_generator.generate_hashlist(metainfo_path = mPath)

        # Now generate the vt report by querying VirusTotal
        malware_label_generator.get_vt_report(hashList = hashListAllMalware, 
                                            outputFilePath = outputReportPath)

    @staticmethod
    def generate_vt_detection_distribution(VTReportPath):
        """
        Reads the virustotal report and outputs the distribution of the vt detections vs number of applications

        params:
            - VTReportPath: Path of the VT report
        Output:
            - detectionDistribution: Dict with key=vt detection and value= # of apks
        """
        detectionDistribution = {}

        # Get the report
        with open(VTReportPath,"rb") as f:
            vt_rep = json.load(f)

        for hash, vtReport in vt_rep.items():
            numPositives = vtReport['results']['positives']
            
            if numPositives in detectionDistribution:
                detectionDistribution[numPositives] += 1
            else:
                detectionDistribution[numPositives] = 1

        # Sort on the basis of num detections
        detectionDistribution = {k:v for k,v in sorted(detectionDistribution.items(), key = lambda item: item[1], reverse=True)}

        print(f"#Detections\t#Apks")
        for numDetection, numApps in detectionDistribution.items():
            print(f"{numDetection}\t{numApps}")
                
        return detectionDistribution


    @staticmethod
    def read_vt_and_convert_to_avclass_format(infile, outfile, filter_hash_list = None):
        """
        Reads the vt report and converts it into a format that AVClass can process.
        AVClass reades a sequence of reports from VirusTotal formatted as JSON records (one per line)

        params:
            - infile: Path of the vt report that should be converted to the simplified JSON format used by AVClass2 
            - outfile: Path of the output file
            - filter_hash_list: Contains list of hashes that needs to be considered for the conversion to AVClass format
        """
        with open(infile,"rb") as f:
            vt_rep = json.load(f)

        # simplified json format used by avclass {md5, sha1, sha256, av_labels}
        avclass_list = []
        
        for apkHash in filter_hash_list:
            print(f"Processing hash {apkHash}")
            try:
                vt_report = vt_rep[apkHash]
            except:
                print(f"Hash {apkHash} not found in the vt report")
                continue
            
            # Generate av labels for each antivirus
            avclass_avlabel_entry = []
            for av, avReport in vt_report["results"]["scans"].items():
                if avReport["detected"] == True:
                    avclass_avlabel_entry.append([av,avReport["result"]])
            
            # If no av detections then skip this file 
            if avclass_avlabel_entry:
                avclass_entry = {}
                avclass_entry["sha1"] = vt_report["results"]["sha1"]
                avclass_entry["md5"] = vt_report["results"]["md5"]
                avclass_entry["sha256"] = vt_report["results"]["sha256"]
                avclass_entry["av_labels"] = avclass_avlabel_entry
                avclass_list.append(avclass_entry)
    
        # Output the list into a file with one report per line
        with open(outfile,'w') as f:
            f.write("\n".join(map(str,avclass_list)).replace("'",'"'))
            
    @staticmethod
    def extract_family_class_from_line(avreport_line):
        """
        Extracts the family and class information from the avclass line.
        params:
            - avreport_line: Line from the avclass report
        Output:
            - result_dict: Dict with key=hash and value= list of classes/families associated with the hash
        """ 
        result = None
        print(avreport_line)
        # use regex to extract hash, classes, and families
        pattern = re.compile(r'(?P<hash>\w+)\t\d+\t(?P<info>.+)')
        match = pattern.match(avreport_line)
        if match:
            hash_val = match.group('hash')
            info = match.group('info')

            # use regex to extract class and family information
            class_pattern = re.compile(r'CLASS:(?P<class>\w+)(:\w+)*')
            classes = class_pattern.findall(info)

            fam_pattern = re.compile(r'FAM:(?P<family>\w+)\|(?P<freq>\d+)')
            families = fam_pattern.findall(info)

            # create dictionary with extracted information
            output_dict = {'CLASS': [], 'FAM': []}
            for class_item in classes:
                if class_item[1] != '' and (class_item[1].lstrip(':') not in output_dict['CLASS']):
                    output_dict['CLASS'].append(class_item[1].lstrip(':'))
                if class_item[0] not in output_dict['CLASS']:
                    output_dict['CLASS'].append(class_item[0])
                
            for fam_item in families:
                output_dict['FAM'].append(fam_item[0])

            result = {hash_val: output_dict}
        print(result)
        print("-----------------")
        return result
    
    @staticmethod
    def get_family_class_from_AVClass_report(avreport_base_directory):
        """
        Reads the AVClass report and outputs the family and class distribution. 
        Assumes that the avcalss report is already generated (using benign_malware_characterization_table_generator()).
        """
        dataset_names = ["std", "cdyear1", "cdyear2", "cdyear3"]
        parsed_avclass_report_dataset = {}
        
        # Parse the AVClass report for each dataset and track the class and family for each hash
        for dataset_ in dataset_names:
            parsed_avclass_report = {}
            
            # Read the AVClass report and store each line in a list
            with open(os.path.join(avreport_base_directory, f'avclass_report_{dataset_}.txt'), 'r') as file:
                avclass_hashLines = file.readlines()
            
            
            for indx, avclass_line in enumerate(avclass_hashLines):
                parsed_info = malware_label_generator.extract_family_class_from_line(avclass_line)
                if parsed_info is not None:
                    parsed_avclass_report.update(parsed_info)
        
            parsed_avclass_report_dataset[dataset_] = parsed_avclass_report
        
        # Logbook to maintain the family and class distribution for each dataset
        distribution_logbook = {}    
        # Generate the family and class distribution for each dataset
        for dataset_, parsed_avclass_report_ in parsed_avclass_report_dataset.items():
            distribution_logbook[dataset_] = {"CLASS": {}, "FAM": {}}
            for hash_, hash_info in parsed_avclass_report_.items():
                if hash_info["CLASS"]:
                    for class_ in hash_info["CLASS"]:
                        if class_ in distribution_logbook[dataset_]["CLASS"]:
                            distribution_logbook[dataset_]["CLASS"][class_] += 1
                        else:
                            distribution_logbook[dataset_]["CLASS"][class_] = 1
                if hash_info["FAM"]:
                    for fam_ in hash_info["FAM"]:
                        if fam_ in distribution_logbook[dataset_]["FAM"]:
                            distribution_logbook[dataset_]["FAM"][fam_] += 1
                        else:
                            distribution_logbook[dataset_]["FAM"][fam_] = 1
        
        # Add any missing keys to the dictionary with a value of 0 
        all_keys_class = []
        all_keys_fam = []
        for dataset, dist in distribution_logbook.items():
            all_keys_class += list(dist['CLASS'].keys())
            all_keys_fam += list(dist['FAM'].keys())
        all_keys_class = list(set(all_keys_class))
        all_keys_fam = list(set(all_keys_fam))
        # Iterate over all datasets in the distribution_logbook dictionary
        for dataset in distribution_logbook:    
            # Iterate over all keys and add any missing keys to the dictionary with a value of 0
            for key in all_keys_class:
                if key not in distribution_logbook[dataset]['CLASS']:
                    distribution_logbook[dataset]['CLASS'][key] = 0
            for key in all_keys_fam:
                if key not in distribution_logbook[dataset]['FAM']:
                    distribution_logbook[dataset]['FAM'][key] = 0

        # Sort the logbook based on the frequency of the class and family        
        sorted_logbook = {}
        for dataset, dist in distribution_logbook.items():
            sorted_fam = {k: v for k, v in sorted(dist['FAM'].items(), key=lambda item: item[1], reverse=True)}
            sorted_class = {k: v for k, v in sorted(dist['CLASS'].items(), key=lambda item: item[1], reverse=True)}
            sorted_logbook[dataset] = {"CLASS": sorted_class, "FAM": sorted_fam}

        malware_apk_per_dataset = {"std": 970, "cdyear1": 489, "cdyear2": 451, "cdyear3": 450}
        # Normalize the distribution based on the number of malware apks in each dataset
        for dataset, dist in sorted_logbook.items():
            for key, value in dist['CLASS'].items():
                sorted_logbook[dataset]['CLASS'][key] = int((value / malware_apk_per_dataset[dataset])*100)
            for key, value in dist['FAM'].items():
                sorted_logbook[dataset]['FAM'][key] = int((value / malware_apk_per_dataset[dataset])*100)
                
        # Pretty print the distribution_logbook
        for dataset_, hash_info in sorted_logbook.items():
            print(f"Dataset: {dataset_}")
            for key_, value_ in hash_info.items():
                print(f"{key_}: {value_}")
            print("---------------------------------------------------")
        
        # Save the sorted_logbook as a json file (NOTE: individual values are normalized)
        with open(os.path.join(avreport_base_directory, "malware_characterization.json"), 'w') as file:
            json.dump(sorted_logbook, file, indent=4)
            
            
                    
def benign_malware_characterization_table_generator(xmd_base_folder):
    """
    Generates the table for benign and malware characterization
    params:
        - xmd_base_folder: Base folder of xmd
    """
    output_folder_dataset_characterization = os.path.join(xmd_base_folder, "res", "dataset_characterization")
    if not os.path.exists(output_folder_dataset_characterization):
        os.makedirs(output_folder_dataset_characterization)

    final_dataset_benignApp_category_info = {} 
    dataset_names = ["std", "cdyear1", "cdyear2", "cdyear3"]
    
    # Get the hashlist of benign and malware for the final dataset. List is generated by generate_apk_list_for_software_AV_comparison() in create_raw_dataset.py.
    for dataset in dataset_names:
        final_dataset_benignApp_category_info[dataset] = {}

        # for apk_type in ["benign","malware"]:
        for apk_type in ["malware"]:
            print(f"Processing {dataset} {apk_type}...")
            final_dataset_benignApp_category_info[dataset][apk_type] = {}

            # Load a pikle file containing the hashlist of benign and malware
            hashlist_path = os.path.join(xmd_base_folder, "res", "APK_hashList_final_dataset_no_filter", f"{dataset}-dataset_{apk_type}HashList.pkl")
            with open(hashlist_path, "rb") as f:
                hashlist = pickle.load(f)

            if apk_type == "benign":
                # Read the app info json file containing app info for each benign apk
                with open("/data/hkumar64/projects/arm-telemetry/xmd/baremetal_data_collection_framework/androzoo/top_app_androzoo_info.json", "rb") as f:
                    app_info = json.load(f)
                
                # Get the category for each benign apk and update the final_dataset_benignApp_category_info dict
                for benignHash in hashlist:
                    app_category = app_info[benignHash]["category"]
                    if app_category in final_dataset_benignApp_category_info[dataset][apk_type]:
                        final_dataset_benignApp_category_info[dataset][apk_type][app_category] += 1
                    else:
                        final_dataset_benignApp_category_info[dataset][apk_type][app_category] = 1

            elif apk_type == "malware":
                # Generate the modified virus total report for each dataset which will be fed to AVClass
                malware_label_generator.read_vt_and_convert_to_avclass_format(infile="/data/hkumar64/projects/arm-telemetry/KUMal/res/virustotal/hash_VT_report_all_malware_vt10.json", 
                                                                            outfile=os.path.join(output_folder_dataset_characterization, f"{dataset}_{apk_type}_avclass.vt"), 
                                                                            filter_hash_list = hashlist)
    # Sort final_dataset_benignApp_category_info dict based on the app category
    # Extract the categories from the dictionary
    categories = set()
    for dataset in final_dataset_benignApp_category_info.values():
        categories |= set(dataset['benign'].keys())
    # Sort the categories alphabetically
    sorted_categories = sorted(categories)
    # Create a new dictionary with sorted categories
    sorted_dataset = {}
    for key, dataset in final_dataset_benignApp_category_info.items():
        sorted_dataset[key] = {
            'benign': {cat: dataset['benign'].get(cat, 0) for cat in sorted_categories},
            'malware': dataset['malware']
        }

    # Save the final_dataset_benignApp_category_info dict as a json file
    with open(os.path.join(output_folder_dataset_characterization, "final_dataset_benignApp_category_info.json"), "w") as f:
        json.dump(sorted_dataset, f, indent=4)
            
    
def main():
    # Current directory [where the script is executing from]
    cur_path = os.path.dirname(os.path.realpath(__file__))
    
    # Base folder of xmd
    xmd_base_folder = cur_path.replace("/src","")
    
    # Folder storing the metaInfo files
    metaInfoPath = os.path.join(xmd_base_folder, "baremetal_data_collection_framework", "androzoo", "metainfo")

    # Path where the final vt report will be saved
    vtReportSavePath = os.path.join(xmd_base_folder,"res","virustotal","hash_VT_report_all_malware_vt10.json")

    # # Generate the VT report
    # eParse = malware_label_generator()
    # eParse.generate_vt_report_all_malware(metaInfoPath = metaInfoPath, outputReportPath = vtReportSavePath)
    
    # # Generate the benign and malware characterization table
    # benign_malware_characterization_table_generator(xmd_base_folder)
    malware_label_generator.get_family_class_from_AVClass_report(avreport_base_directory="/data/hkumar64/projects/arm-telemetry/KUMal/res/dataset_characterization/avclass_reports")
    exit()
    # Get the detection distribution
    eParse.generate_vt_detection_distribution(VTReportPath="/data/hkumar64/projects/arm-telemetry/xmd/res/virustotal/hash_VT_report_all_malware.json")
    ########################################## Generating VT report for feeding to AVClass ################################################################
    # eParse.read_vt_and_convert_to_avclass_format(infile= "/data/hkumar64/projects/arm-telemetry/xmd/res/virustotal/hash_virustotal_report_malware", 
    #                                             outfile="/data/hkumar64/projects/arm-telemetry/xmd/res/virustotal/avclass_virustotal_report_malware.vt")
    #######################################################################################################################################################
if(__name__=="__main__"):
	main()

