"""
Python script to extract the meta info from the parser info files
"""

import os
import json
import re
from utils import malware_label_generator

def parse_json(json_path):
    """
    Reads the json file and returns the parsed json object
    """
    with open(json_path, "r") as f:
        json_obj = json.load(f)
    
    parsed_json_obj = {}
    for filePath, iter_rn_list in json_obj.items():
        # Extract the hash from the filePath
        hash_pattern = r'(?<=_)[A-F0-9]{64}(?=\.)'
        match = re.search(hash_pattern, filePath)
        if match:
            hash_value = match.group().strip()
            # print("Output =", hash_value)
            parsed_json_obj[hash_value] = {}
        else:
            print(filePath)
            raise(ValueError("Hash not found in the file path"))

        for iter_rn_entry in iter_rn_list[1:]:
            key = list(iter_rn_entry.keys())[0]
            value = list(iter_rn_entry.values())[0]
            
            iter_rn_pattern = r"_iter_(\d+)_rn(\d+)\."
            match = re.search(iter_rn_pattern, key)
            if match:
                iter_value = int(match.group(1).strip())
                rn_value = int(match.group(2).strip())
                # print(f"iter={iter_value}, rn={rn_value}")
                if iter_value not in parsed_json_obj[hash_value]:
                    parsed_json_obj[hash_value][iter_value] = {rn_value: value}
                else:
                    parsed_json_obj[hash_value][iter_value][rn_value] = value
                
            else:
                print(iter_rn_entry)
                raise(ValueError("Iter and RN not found in the iter rn entry"))
         
    return parsed_json_obj
        
def get_dataset_metainfo(dataset_name, base_dir_path):
        """
        Function to get the dataset meta info
        params:
            - dataset_name: Name of the dataset [std-dataset, cdyear1-dataset, cdyear2-dataset, cdyear3-dataset]
            - base_dir_path: Path to the base directory of the project
        
        Output:
            - parsed_json_obj: Parsed json object
            To access an entry in the parsed json object, use the following syntax:
                parsed_json_obj[hash_value][iter_value][rn_value] = [freq_logcat_event_per_file, num_logcat_lines_per_file, runtime_per_file]
        """
        
        if dataset_name == "std-dataset":
            # Get the location of the parser info files
            parser_info_benign1 = os.path.join(base_dir_path, "res/parser_info_files/kumal", f"parser_info_std_benign.json")
            parser_info_benign2 = os.path.join(base_dir_path, "res/parser_info_files/kumal", f"parser_info_std_benign_dev2.json")
            parser_info_malware1 = os.path.join(base_dir_path, "res/parser_info_files/kumal", f"parser_info_std_malware.json")
            parser_info_malware2 = os.path.join(base_dir_path, "res/parser_info_files/kumal", f"parser_info_std_vt10_malware_dev2.json")
            
            # Parse the json files
            parsed_json_obj1 = parse_json(parser_info_benign1)
            parsed_json_obj2 = parse_json(parser_info_benign2)
            parsed_json_obj3 = parse_json(parser_info_malware1)
            parsed_json_obj4 = parse_json(parser_info_malware2)
            
            # Merge the parsed json objects
            parsed_json_obj = {**parsed_json_obj1, **parsed_json_obj2, **parsed_json_obj3, **parsed_json_obj4}
            return parsed_json_obj
        
        elif dataset_name == "cdyear1-dataset":
            # Get the location of the parser info files
            parser_info_benign = os.path.join(base_dir_path, "res/parser_info_files/kumal", f"parser_info_cd_year1_benign.json")
            parser_info_malware = os.path.join(base_dir_path, "res/parser_info_files/kumal", f"parser_info_cd_year1_malware.json")

            # Parse the json files
            parsed_json_obj1 = parse_json(parser_info_benign)
            parsed_json_obj2 = parse_json(parser_info_malware)
            parsed_json_obj = {**parsed_json_obj1, **parsed_json_obj2}
            return parsed_json_obj
        
        elif dataset_name == "cdyear2-dataset":
            # Get the location of the parser info files
            parser_info_benign = os.path.join(base_dir_path, "res/parser_info_files/kumal", f"parser_info_cd_year2_benign.json")
            parser_info_malware = os.path.join(base_dir_path, "res/parser_info_files/kumal", f"parser_info_cd_year2_malware.json")

            # Parse the json files
            parsed_json_obj1 = parse_json(parser_info_benign)
            parsed_json_obj2 = parse_json(parser_info_malware)
            parsed_json_obj = {**parsed_json_obj1, **parsed_json_obj2}
            return parsed_json_obj
        
        elif dataset_name == "cdyear3-dataset":
            # Get the location of the parser info files
            parser_info_benign = os.path.join(base_dir_path, "res/parser_info_files/kumal", f"parser_info_cd_year3_benign.json")
            parser_info_malware = os.path.join(base_dir_path, "res/parser_info_files/kumal", f"parser_info_cd_year3_malware.json")
            
            # Parse the json files
            parsed_json_obj1 = parse_json(parser_info_benign)
            parsed_json_obj2 = parse_json(parser_info_malware)
            parsed_json_obj = {**parsed_json_obj1, **parsed_json_obj2}
            return parsed_json_obj
                    
        else:
            raise(ValueError("Incorrect dataset type specified"))

# def get_malware_hashClassFamily_dict(dataset_name, base_dir_path):
#     """
    
#     """
#     avreport_base_directory= os.path.join(base_dir_path, "res/dataset_characterization/avclass_reports")
#     parsed_avclass_report_dataset, top_family_class_logbook = malware_label_generator.get_family_class_from_AVClass_report(avreport_base_directory)
    
#     dataset_name = os.replace(dataset_name, "-dataset", "")
#     hashClassFamily_dict = parsed_avclass_report_dataset[dataset_name]
#     top_family_class_logbook = top_family_class_logbook[dataset_name]
        
def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    base_folder_location = os.path.join(dir_path.replace("/src",""),"")

    parsed_json_obj = get_dataset_metainfo("std-dataset", base_folder_location)
        
if __name__ == '__main__':
    main()