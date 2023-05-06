import sys
import os
import time
import sh
import re
# Drives main_simpleperf.py and main_dvfs_channel.py and the network data collection module.


# Routine to extract the package name from the apk file name
def extract_package_name(apk_file_name):
	package_string = sh.grep(sh.aapt('dump','badging', apk_file_name), 'package:\ name')
	packageObj = re.match( r'package:\ name=\'([^\']*)\'', str(package_string), re.M|re.I)

	if packageObj:
		package_name = packageObj.group(1)
	else:
		sys.exit("Package name cannot be extracted")

	return package_name


def main():

	#sys_arguments for this wrapper
	run=sys.argv[1]
	adb_pull=sys.argv[2]
	remove=sys.argv[3]
	apk_file_name=sys.argv[4]   #Name of the apk file (along with path), not the package name
	dest_folder=sys.argv[5] # This is the top directory in which all the results will be stored
	samp_time_perf=sys.argv[6] # This is in ms
	run_time=sys.argv[7]
	iter_number=sys.argv[8]
	
	# Extract the package name from the apk file
	package = extract_package_name(apk_file_name)

	print("All ARGS obtained")

	#************************* Set of arguments for main_simpleperf.py *************************#
	# arg1 is enable for run | arg2 is enable for adb_pull 
	# arg3 is enable for removing log file  
	# arg4 is the package name to be monitored
	run_perf = run
	adb_pull_perf = adb_pull
	remove_perf= remove
	package_name_perf = package
	
	# destination folder at the laptop
	# Each package will have its own directory, and within each package's directory: each channel (perf, dvfs, network) will have its own sub-directory
	
	# Top directory for results
	os.system("mkdir -p "+dest_folder)
	
	# Directory for the package
	package_dir_name = package.replace(".","_")+"_"+apk_file_name.split("/")[-1]
	os.system("mkdir -p "+dest_folder+"/"+package_dir_name)

	# Directory for simpleperf
	os.system("mkdir -p "+dest_folder+"/"+package_dir_name+"/simpleperf")
	# arg5 is the destination folder in the laptop
	dest_folder_perf=dest_folder+"/"+package_dir_name+"/simpleperf"
	
	# arg6: This is the sampling interval (in ms)
	samp_time_perf=samp_time_perf
	# arg7: Duration for which you want simpleperf to collect the data (in s)
	run_time_perf=run_time
	# arg8: Iteration number [to label the file with the iteration]
	iter_number_perf=iter_number

	
	#************************* Runing both the scripts *************************#
	os.system("python main_simpleperf.py "+run_perf+" "+adb_pull_perf+" "+remove_perf+" "+package_name_perf+" "+dest_folder_perf+" "+samp_time_perf+" "+run_time_perf+" "+iter_number_perf+" &")
	




if(__name__=="__main__"):
	main()