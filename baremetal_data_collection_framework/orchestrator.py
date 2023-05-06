"""
Orchestrator which will direct the entire data collection process
Start App Interaction -> Start Data Collection -> Flash Firmware (If dealing with malware)

Iterate through every application in a specified folder.
For each application run N iterations of data collection where each iteration will perform new action.
You will also have 4 repeated iterations for each of the iteration in N (To collect all the performance counter channels)
"""

import os
import sys
import time
import argparse
from os import listdir
from os.path import isfile, join
import sh
import re
import random
import logging
import dropbox_upload
import threading
from multiprocessing import Process
import firmware_flash_script
import subprocess

#--------------------------------------------------------------------------------------------------------------------------#
# Number of independent iterations of data for each application
num_iter_independent = 2

# Number of same iterations of data for each application
num_iter_dependent = 4 # One for every group of performance counters 
#--------------------------------------------------------------------------------------------------------------------------#

# Time for which to interact with the device
broadcast_event_interaction_time = 20
monkey_interaction_time = 35

# Configuration parameters common for all the apps and iteration
adb_pull = 1
remove = 1

samp_time_perf= 100 # This is in ms

# This is the time for which you will capture telemetry (DVFS and Performance Counter)
run_time= monkey_interaction_time + broadcast_event_interaction_time + monkey_interaction_time

# Number of apk runs with errors, beyond which we abort the apk and move on to the next apk
abort_threshold = 2

############################################################## System Broadcast Events ##############################################################
sys_broadcast_event = ['BOOT_COMPLETED','SMS_RECEIVED','SCREEN_ON','WAP_PUSH_RECEIVED','CONNECTIVITY_CHANGE','PICK_WIFI_WORK','PHONE_STATE',
'UMS_CONNECTED','UMS_DISCONNECTED','ACTION_MAIN','PACKAGE_ADDED','PACKAGE_REMOVED','PACKAGE_CHANGED','PACKAGE_REPLACED','PACKAGE_RESTARTED',
'PACKAGE_INSTALL','ACTION_POWER_CONNECTED','ACTION_POWER_DISCONNECTED', 'BATTERY_LOW','BATTERY_OKAY','BATTERY_CHANGED_ACTION','USER_PRESENT',
'INPUT_METHOD_CHANGED','SIG_STR','SIM_FULL']
#####################################################################################################################################################

def get_app_list(path_dir): 
	"""
	Returns the list of files in a directory.
	params:
		- path_dir: Path of the directory for which the file-list should be generated
	Output:
		- file_name: List of files contained in the directory specified by path_dir 
	"""
	## NOTE: Make sure that file names do not have spaces, else the entire script crashes.
	file_names = [f for f in listdir(path_dir) if isfile(join(path_dir, f))]
	return file_names


def extract_package_name(apk_file_name):
	"""
	Function to extract the package name from the apk file name.
	params:
		- apk_file_name: apk file name along with path of the apk

	Output:
		- package_name : String if package name found, else 0.
	"""
	print(f" - Extracting the package name from : {apk_file_name}")

	try:
		package_string = sh.grep(sh.aapt('dump','badging', apk_file_name), 'package:\ name')

	# Error raised in retrieving the package name. Return 0 and check for this when the package name is returned.
	except sh.ErrorReturnCode: 
		return 0

	packageObj = re.match( r'package:\ name=\'([^\']*)\'', str(package_string), re.M|re.I)

	if packageObj:
		package_name = packageObj.group(1)
	else:
		# Package name was not extracted
		return 0 

	return package_name


def extract_activity_name(apk_file_name):
	"""
	Function to extract the launchable activity name from the apk file name.
	params:
		- apk_file_name: apk file name along with path of the apk

	Output:
		- activity_name : String if activity name found, else 0.
	"""
	print(f" - Extracting the activity name from : {apk_file_name}")

	try:
		activity_string = sh.grep(sh.aapt('dump','badging', apk_file_name), 'activity')

	# Error raised in retrieving the package name. Return 0 and check for this when the activty name is returned.
	except sh.ErrorReturnCode: 
		return 0

	activityObj = re.match( r'name=\'([^\']*)\'', str(activity_string), re.M|re.I)

	if activityObj:
		activity_name = activityObj.group(1)
	else:
		# Activity name was not extracted
		return 0 

	return activity_name

def interact_with_benchmark_app(package_name):
	"""
	Starts the benchmark application and runs it.
	"""
	print("****************** STARTING INTERACTION WITH BENCHMARK ******************")
	ignore_this_apk=0
	# Start the apk
	os.system(f"adb shell monkey -p {package_name} -c android.intent.category.LAUNCHER 1")

	# Click the ok button [to skip the warning]
	time.sleep(1)
	os.system(f"adb shell input tap 900 1269")

	# Get the location of the run button and click on that
	os.system("adb pull $(adb shell uiautomator dump | grep -oP '[^ ]+.xml') /tmp/view.xml")
	time.sleep(1)
	
	command = "echo $(perl -ne 'printf \"%d %d\n\", ($1+$3)/2, ($2+$4)/2 if /text=\"Run..\"[^>]*bounds=\"\\[(\\d+),(\\d+)\\]\\[(\\d+),(\\d+)\\]\"/' /tmp/view.xml)"
	get_coord = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout
	get_coord_string = get_coord.read().decode()

	time.sleep(1)
	os.system(f"adb shell input tap {get_coord_string}")
	
	# Remove the tmp file
	time.sleep(1)
	os.system("rm /tmp/view.xml")

	# Allow the app to run for some time
	time.sleep(15) 

	return ignore_this_apk

def interact_with_app_touch_events(package_name, activity_name, r_time, seed, log_file_handler, ignore_this_apk):
	"""
	Function to perform monkey tool based interaction.

	params:
		- package_name : Package with which we want to interact with.
		- activity_name : Top activity
		- r_time : Time for which we want monkey to interact with the device [in seconds]
		- seed : seed for generating the set of random touch events. If seed is same, then touch events are same.
		- log_file_handler: Handler for the log file
		- ignore_this_apk: Used for tracking how many times the apk process doesn't exist after monkey interaction.

	Output:
		- ignore_this_apk: Updated ignore_this_apk
	"""
	
	print("******* MONKEY INTERACTION STARTING *******")
	# The number doesn't matter because we are using timeout to terminate monkey
	num_events = 2000 

	# Time when monkey started interacting with the app
	s_fuzz_t = time.time() 

	## command to insert UI touches events with 100ms delay
	# pct-majornav : Percentage of major navigation events = 0 so that you do not exit the app 
	monkey_cmd_1 = 'monkey -s '+str(seed)+' --throttle 2000 --pct-majornav 0 --kill-process-after-error -p '+str(package_name)+' ' 
	
	## number of UI touch events
	monkey_cmd_2 = str(num_events)

	# Ask monkey to interact with the app with the specified arguments
	subprocess.Popen('adb shell timeout '+str(r_time)+ " "+monkey_cmd_1+monkey_cmd_2, shell=True).wait()
	
	# Time when monkey ended interaction with the app
	e_fuzz_t = time.time() 

	# Check if the interacion time was less than your desired run time
	# If this is the case, then your app might have crashed 
	if (e_fuzz_t - s_fuzz_t) < r_time:  
		print(f'error')
		log_file_handler.error("Application terminated before run time elapsed : " + str(package_name) + "| Execution time: " + str(e_fuzz_t - s_fuzz_t))

	# Check if a non-zero pid exists. If yes then the application process is running (background or foreground)
	command = "adb shell pidof "+package_name
	get_pid_raw = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout
	get_pid_string = get_pid_raw.read().decode()

	# If no pid is returned, then your application process doesn't exist. Log this.
	if get_pid_string == '': 
		ignore_this_apk += 1

		log_file_handler.error("Application is absent after MONKEY completed: " + str(package_name))
		# Try launching the main activity of the app again 
		if activity_name!=0: # i.e. activity has been successfully extracted
			os.system("adb shell am start -n " + package_name + "/" + activity_name)

	print("******* MONKEY INTERACTION ENDING *******")

	return ignore_this_apk		

def interact_with_app_broadcast_events(package_name, r_time, log_file_handler, ignore_this_apk):
	"""
	Function to send broadcast events to the app (expect malwares to have broadcast receivers)

	params:
		- package_name: Package with which we want to interact with.
		- r_time: Time for which we want monkey to interact with the device [in seconds]
		- log_file_handler:  Handler for the log file
		- ignore_this_apk: Used for tracking how many times the apk process doesn't exist after broadcast interaction.

	Output:
		- ignore_this_apk: Updated ignore_this_apk
	"""

	print("******* BROADCAST INTERACTION STARTING *******")
	
	s_t = time.time()
	# Broadcast events for a duration specified by r_time
	while((time.time()-s_t) < r_time): 
		for j in range(len(sys_broadcast_event)):
			# Sending broad events to specific package provided by -p package_name
			subprocess.Popen('adb shell \"su -c \' am broadcast -a android.intent.action.'+str(sys_broadcast_event[j])+' -p '+package_name+"\'\"", shell=True).wait()

			# Delay
			time.sleep(0.2)

	# Check if a non-zero pid exists. If yes then the application process is running (background or foreground)
	command = "adb shell pidof "+package_name
	get_pid_raw = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout
	get_pid_string = get_pid_raw.read().decode()

	if get_pid_string == '': # If no pid is returned, then your application process doesn't exist. Log this.
		log_file_handler.error("Application process is absent after BROADCAST completed: " + str(package_name))
		ignore_this_apk += 1

	print("******* BROADCAST INTERACTION ENDING *******")

	return ignore_this_apk

def launch_logcat(package_name, log_file_handler, dest_folder, iter_number, run, apk_file, ignore_this_apk):
	"""
	Checks if the package can be started, launches the apk, and then launches logcat.
	params:
		- package_name :  Name of the package for which you want to collect the data.
		- log_file_handler : For writing to the log file.
		- dest_folder : top directory where results are stored for this class of malware.
		- iter_number, run : Iteration and run number for labelling the file.
		- runtime : time for which you want to collect the logcat logs [this is a global variable].
		- ignore_this_apk : Used for tracking how many times this apk has not executed.
	
	Output:
		- ignore_this_apk : Updated ignore_this_apk.
		- log_proc: Handler for the logcat subprocess.
	"""
	print("******* STARTING LOGCAT ROUTINE *******")

	# Launch the apk
	os.system('adb shell monkey -p '+ package_name +' -c android.intent.category.LAUNCHER 1')

	# Check if the apk has been launched and the pid can be extracted (for logcat)
	command = "adb shell pidof "+package_name
	get_pid_raw = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout
	get_pid_string = get_pid_raw.read().decode()
	get_pid_string = get_pid_string.replace("\n","") # Delete the \n at the end of the pid

	# If no pid is returned, then your application process doesn't exist. Log this.
	if get_pid_string == '': 
		# You cannnot launch logcat
		log_file_handler.error("PID is not present for logcat: " + str(package_name))
		ignore_this_apk += 1
		log_proc = None
			
	else: # Launch logcat in the background
		# Used for accessing the folder of respective apks
		package_dir_name = package_name.replace(".","_")+"_"+apk_file 
		# Dest folder where all the logcat files are stored for a particular package
		dest_folder_logcat= dest_folder+"/"+package_dir_name+"/logcat" 
		
		log_proc = subprocess.Popen(f"python main_logcat.py {str(run)} 1 1 {dest_folder_logcat} {package_name} {str(iter_number)} {str(run_time)} {get_pid_string}", shell=True, stdout=subprocess.DEVNULL)
				
	return ignore_this_apk, log_proc


def install_apk(apk_path, apk_name):
	"""
	Function to install the apk on the client device.
	
	Output:
		- inst_apk_proc: Handler for the install apk process

	""" 
	print(f" - Installing the apk : {apk_name}")
	install_apk_proc =  subprocess.Popen(f"adb install -g {apk_path + apk_name}", shell=True)
	return	install_apk_proc

def uninstall_apk(package_name):
	"""
	Function to uninstall the app on the test device
	""" 
	subprocess.Popen(f"adb uninstall {package_name}", shell=True, stdout=subprocess.PIPE).wait()


# Module for creating the log file
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def setup_logger(name, log_file, level=logging.WARNING):
    """
	To setup the logger.
	"""
    handler = logging.FileHandler(log_file, mode = 'a')        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def push_to_dropbox(app_folder_path, dest_folder_path, remove, file_flag, apk_file_path):
	"""
	Module to push the folder/file containing the data logs to dropbox.
	params:
		- app_folder_path : The folder that you want to upload to dropbox
		- dest_folder_path : The path in dropbox where you want to upload the folder to. 
		- remove: Flag for removing the folder on your local machine. 
		- file_flag : Indicates that you want to upload a file and not a folder
		- apk_file_path : The path to the apk file being simulated
	"""
	
	# Fetch the api token
	path_to_token = "/home/harshit/research/arm_data_collection/dropbox_access_token.txt"
	f = open(path_to_token, "r")
	token = f.read() 

	if file_flag: 
		# You want to upload a file
		print("Uploading the log file")
		dropbox_upload.upload(token, app_folder_path, dest_folder_path)

	else: 
		# You want to upload a folder
		print("Uploading the folder")
		dropbox_upload.upload_folder(app_folder_path, app_folder_path, dest_folder_path , token)

	if remove:
		# After the upload is complete, remove the folder or file
		os.system("rm -r "+app_folder_path)

		#Remove the apk_file as well [This is soft-checkpointing, where if we fire the script again, the apks which were run won't be executed again]
		os.system("rm -r "+ apk_file_path)

		# Making a directory (So that the commands following this statement don't throw error)
		os.system('mkdir '+app_folder_path.split("/")[0]+'/temp')

		#Some of the apps folders are not deleted so using an easy fix to just delte the folders not the file lists
		os.system("rm -R `ls -1 -d "+app_folder_path.split("/")[0]+"/*/`")



"""
Module to extract the checksum of the partitions
key = Partition name
value = [partition-location, checksum-before-malware, checksum-after-malware]
"""
partition_details = {
'system_a': ['/dev/block/sda5', None, None],
'system_b': ['/dev/block/sda6', None, None],
'product_a': ['/dev/block/sda7', None, None],
'vendor_a': ['/dev/block/sda9', None, None],
'boot_a': ['/dev/block/sda11', None, None],
'dtbo_a': ['/dev/block/sde11', None, None],
'vbmeta_a': ['/dev/block/sde10', None, None]
}

def extract_partition_checksum(flag_before_malware):
	"""
	Method to populate the entries of the partition_details dict (global variable)
	params:
		- flag_before_malware:
						If flag_before_malware == 1, then checksum-before-malware field is updated with the hash [this is the hash before you installed the malware]
						If flag_before_malware == 0, then checksum-after-malware field is updated with the hash [this is the hash after you have ran the malware]

	Output:
		Updates the partition_details dict.
	"""
	global partition_details
	
	print("******************************** EXTRACTING PARTITION CHECKSUMS ********************************")
	for key,_ in partition_details.copy().items():
		partition = partition_details[key][0]
		command = "adb shell \"su -c \'sha256sum "+ partition +"\'\""

		get_hash_raw = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout

		hash_raw_string = get_hash_raw.read().decode()

		# Using regex to extract the hash
		hashObj = re.match( r'(.*)  ', str(hash_raw_string), re.M|re.I)

		if hashObj:
			hash_str = hashObj.group(1)
			# Store the partition hash in the dict

			# If flag_before_malware is 1, then the checksum is the checksum before infection has happened [store the check sum in checksum-before-malware]
			if flag_before_malware:
				partition_details[key][1] = hash_str
			
			# If flag_before_malware is 0, then the checksum is the checksum after the malware has been installed and ran [store the checksum in checksum-after-malware]
			else:
				partition_details[key][2] = hash_str 	
		else:
			print(key, hash_raw_string)
			print("Hash cannot be extracted")


def check_partition_modification():
	"""
	Method to check whether the system partitions have been modified.
		This method checks whether the two fields of partition_details dict : checksum-before-malware and checksum-after-malware are the same
		If the fields are different, then you have to perform a complete firmware flash
		If the fields are same, then we only flash the data partition
	"""
	# Stores the partitions that have been modified
	modified_partitions = [] 

	for key,value in partition_details.items():
		if (value[1] != value[2]):
			# Partition has been modified
			modified_partitions.append(key)

	return modified_partitions 		

def check_low_battery_power():
	"""
	Method to check if there is enough battery power in the device for data collection.
	Output:
		- low_power_flag : If True, then the device battery is below 15%
	"""
	low_power_flag=False

	command = "adb shell dumpsys battery | grep 'level:' | sed 's/^.*: //'"
	get_power_raw = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout
	time.sleep(1)
	power_raw_string = get_power_raw.read().decode()
	print(f" ----- Current battery level : {power_raw_string} ----- ")

	if (int(power_raw_string) < 15):
		# The purpose of this routine is to give the device some time to recharge [and reduces manual intervention]
		print("############################## BATTERT STATS ##############################")
		subprocess.Popen('adb shell dumpsys battery',shell=True).wait()
		print("###########################################################################")

		low_power_flag=True	

	return low_power_flag

def check_screen_status():
	"""
	Method to check the status of the device screen.

	Output:
		- screen_status : Can be one of the following:
							- OFF_LOCKED: Device screeen is off and locked
							- ON_LOCKED: Device screen is on and locked
							- ON_UNLOCKED: Device screeen is on and unlocked
	"""
	screen_status = None

	command = "adb shell dumpsys nfc | grep \'mScreenState=\'"
	get_disp_status = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout
	disp_status_string = get_disp_status.read().decode().replace("\n","")
	# Using regex to extract the hash
	dispObj = re.match( r'.*mScreenState=(.*)', str(disp_status_string), re.M|re.I)

	if dispObj:
		screen_status = dispObj.group(1)
	
	return screen_status

def reboot_device():
	"""
	Reboots the device and waits till the reboot process has finished.
	"""
	# Reboot 
	os.system('adb reboot')
	time.sleep(20)

	# Check if the device has booted up
	while not firmware_flash_script.check_device_boot_up():
		print(f"[{firmware_flash_script.get_local_time()}] Waiting for the device to boot up.")
		time.sleep(4)

def reset_device():
	"""
	Stops gnirehtet, reboots device, and starts gnirehtet back. Method is used when the device stops responding.
	"""
	# Stop gnirehtet
	subprocess.Popen('gnirehtet stop', shell=True).wait()
	time.sleep(2)

	reboot_device()
	
	# Wait for services to start up
	time.sleep(6)

	# Start gnirehtet again
	subprocess.Popen('gnirehtet start', shell=True).wait()
	time.sleep(2)

def setup_wakeup_device():
	"""
	Method to wakeup the device before the data-collection and the apk execution starts
	"""
	unlock_count = 0
	print(" - Waking up and setting up the device for data-collection.")
	
	try:
		# Keep on repeating the action until the device screen is ON and unlocked
		while check_screen_status() != "ON_UNLOCKED":
			
			print(f"  - Attempting to unlock and setup the phone : attempt-{unlock_count}")
			
			if check_screen_status() == "OFF_LOCKED":
				# Keep on waking up the device until the screen is on 
				while check_screen_status() != "ON_LOCKED":
					subprocess.Popen('adb shell input keyevent KEYCODE_WAKEUP', shell=True).wait(timeout=60)
					time.sleep(0.5)

			subprocess.Popen('adb shell input keyevent KEYCODE_WAKEUP', shell=True).wait(timeout=60)
			time.sleep(0.5)
			subprocess.Popen('adb shell input keyevent KEYCODE_WAKEUP', shell=True).wait(timeout=60)
			time.sleep(0.5)

			# Swipe up to move away from the lock screen
			subprocess.Popen('adb shell input swipe 500 1000 300 300', shell=True).wait(timeout=60)
			time.sleep(1)
			
			# Go to home screen (if any other app is opened)
			subprocess.Popen('adb shell am start -a android.intent.action.MAIN -c android.intent.category.HOME', shell=True).wait(timeout=60)
			time.sleep(1)	

			unlock_count+=1

			if unlock_count > 20: 
				# If the device is stuck, then reboot the device.
				print(f"  - Unlock attempts exceeded. Rebooting the device.")

				# Rebooting the device
				reset_device()

				# Unlock and wake up device [Keeps on repeating the routine until the device status is ON and UNLOCKED]
				setup_wakeup_device()

	except subprocess.TimeoutExpired:
		print(" - Timeout exception raised. Rebooting the device. ")
		# If any of the command times out, then try resetting the device
		reset_device()
		# Unlock and wake up device [Keeps on repeating the routine until the device status is ON and UNLOCKED]
		setup_wakeup_device()

	print(" - Screen ON and UNLOCKED. Device ready for data collection.")

def setup_wakeup_device_process():
	"""
	Launches the setup_wakeup_device process and monitors for timeouts.
	"""
	setup_completion_flag = False

	while setup_completion_flag != True:
	
		setup_process = Process(target=setup_wakeup_device, name="test_timeout")
		setup_process.start()
		
		# Wait for the setup to finish. If the setup takes longer than 10 mins, then we're probably stuck. 
		setup_process.join(timeout=600)
		
		# Check if the process has sucessfully executed or timed out. Exitcode is 0 only when the process finishes properly.
		if setup_process.exitcode != 0:
			print("*********** Setup process has timed out. Terminating and rebooting the device. ***********")
			# Process has timed out. Terminate the process and reset the device.
			setup_process.terminate()
			
			# Reboot the device
			reboot_device()

			# Wait for services to start up
			time.sleep(10)

			# Start gnirehtet again
			subprocess.Popen('gnirehtet start', shell=True).wait()
			time.sleep(2)


		# If the setup process is successfully finished
		elif setup_process.exitcode == 0: 
			setup_completion_flag = True

	print(" - Setup process complete.")

def collect_data(path_dir, MALWARE_FLAG, dest_folder, log_file_handler, app_type):
	"""
	Module to orchestrate the collection of the data logs.
	params:
		- path_dir : Path of the directory containg the applications
		- MALWARE_FLAG : Flag to indicate if the directory contains malware applications 
						(if malware flag is passed, then we uninstall and reinstall the apk after every iteration)
		- dest_folder : Destination folder where the collected data is stored [top directory in which all the results will be stored]
		- log_file_handler : Handler of the log file 
		- app_type : experiment name : used for tracking the dataset
	"""
	# Used for identifying whether we're dealing with malware
	# If malware, then you will uninstall and reinstall after every iteration
	BENIGN_FLAG = 1-MALWARE_FLAG

	# Get the apk file names from the virustotal report
	file_names = get_app_list(path_dir)

	# Start gnirehtet (NOTE: Make sure that the relay server is started in a separate terminal)
	subprocess.Popen('gnirehtet start', shell=True).wait()
	time.sleep(2)

	if MALWARE_FLAG:
		# Populate the checksum-before-malware field of all the partitions, because your device is not infected right now. 
		extract_partition_checksum(flag_before_malware= 1)

	# Wake up and setup the device
	setup_wakeup_device_process()

	# Iterate through each apk file in the directory
	for apk_file in file_names:

		# Go to sleep if the device doesn't have enough battery left
		if check_low_battery_power():
			print('************************* Battery level below 15% : Going to sleep for 2.5 hour *************************')
			time.sleep(9000)
		
		#This is the full path to the apk file being used
		apk_file_path = os.path.join(path_dir,apk_file)
		
		# If any of these flags > abort_threshold, then we do not run this app
		# These flags are set by the functions : launch_logcat(), interact_with_app_touch_events(), interact_with_app_broadcast_events()
		ignore_this_apk_logcat = 0
		ignore_this_apk_monkey = 0
		ignore_this_apk_broadcast = 0

		# Extract the package name from the apk file name
		package_name = extract_package_name(apk_file_name=apk_file_path)
		# Extract the launchable activity name from the apk file name
		activity_name = extract_activity_name(apk_file_name=apk_file_path)

		# Check if the package name has been successfully extracted. If not, then skip this apk and log this
		if package_name == 0 : # Error in retrieving the package
			# Log this apk
			log_string = f"PACKAGE NAME NOT EXTRACTED ----> Apk name: {apk_file}"					
			log_file_handler.error(log_string)
			print(" - SKIPPING THE APK: UNSUCCESSFUL PACKAGE EXTRACTION")
			# Skip this apk file
			continue

		# Check if the activity name has been successfully extracted, If not, then log this
		if activity_name == 0: # Error in retrieving the activity name
			# Log this
			log_string = f"ACTIVITY NAME NOT EXTRACTED ----> Apk name: {apk_file}"			
			log_file_handler.error(log_string)

		# Used for creating the folder of respective apks
		package_dir_name = package_name.replace(".","_")+"_"+apk_file 
		
		######################################## Starting an independent iteration ########################################

		for ind_iter in range(num_iter_independent):

			# Check the ignore_this_apk flags
			if((ignore_this_apk_logcat > abort_threshold) or (ignore_this_apk_monkey > abort_threshold) or (ignore_this_apk_broadcast > abort_threshold)):
				log_string = f"Skip independent iteration Apk name: {apk_file} | Package name: {package_name} | Iteration: ({str(ind_iter)},{str(dep_iter+1)})"
				log_file_handler.warning(log_string)
				break

			# Generate a new seed for every independent iteration
			seed = random.randint(1,10000) # Seed is a random number between 1 and 10000 
			print(f"-Starting a new independent iteration : {ind_iter}")
					
			for dep_iter in range(num_iter_dependent):
				
				# Check the ignore_this_apk flags
				if((ignore_this_apk_logcat > abort_threshold) or (ignore_this_apk_monkey > abort_threshold) or (ignore_this_apk_broadcast > abort_threshold)):
					log_string = f"Skip dependent iteration Apk name: {apk_file} | Package name: {package_name} | Iteration: ({str(ind_iter)},{str(dep_iter+1)})"
					log_file_handler.warning(log_string)
					break

				print(f'-Dependent iteration {dep_iter+1}')	

                # Install the apk (benign or malware)
				iApkProc = install_apk(path_dir, apk_file)
				# If unable to install this apk then move to the next iteration
				try:	
					iApkProc.wait(timeout=60)
				except subprocess.TimeoutExpired:
					iApkProc.kill()
					continue
				
				################################################### Setting up parameters and device for data collection ############################################################	
				# run: used for selecting the group of performance counters (and labelling the files for consistency)
				run = str(dep_iter+1)
				# iter_number: used for labelling the iteration
				iter_number = str(ind_iter) 
					
				# Log the details of this iteration in the log file
				log_string = "Apk name: "+apk_file+" | " + "Package name: "+package_name+" | "+ "Iteration: ("+str(ind_iter)+","+str(dep_iter+1)+")"+ " | "+"Seed: "+str(seed) +" | "+ "Data Collection Time: "+ str(run_time)+ " | " + "Malware: " +str(MALWARE_FLAG) + " | " + "Benign: " +str(BENIGN_FLAG)
				log_file_handler.warning(log_string)
				
				# Wakeup and setup the device for data-collection process to start
				setup_wakeup_device_process()
				#####################################################################################################################################################################

				################################################### DATA COLLECTION START ############################################################
				dc_proc = subprocess.Popen(f"python3 main_comb_devfreq_simpleperf.py {str(run)} {str(adb_pull)} {str(remove)} {str(apk_file_path)} {str(dest_folder)} {str(samp_time_perf)} {str(run_time)} {str(iter_number)}", shell=True, stdout=subprocess.DEVNULL)
				
				# Launch the apk and start logcat 
				ignore_this_apk_logcat, _ = launch_logcat(package_name, log_file_handler, dest_folder, iter_number, run, apk_file, ignore_this_apk_logcat)
				
				# Simulate monkey
				ignore_this_apk_monkey = interact_with_app_touch_events(package_name, activity_name, monkey_interaction_time, seed, log_file_handler, ignore_this_apk_monkey)
							
				# Start broadcast interaction
				ignore_this_apk_broadcast = interact_with_app_broadcast_events(package_name, broadcast_event_interaction_time, log_file_handler, ignore_this_apk_broadcast)

				# Simulate monkey
				ignore_this_apk_monkey = interact_with_app_touch_events(package_name, activity_name, monkey_interaction_time, seed, log_file_handler, ignore_this_apk_monkey)

				# Buffer period for data collection to be over. Sleep for some time and allow the data to be transferred to the host.
				time.sleep(8)
				################################################### DATA COLLECTION END ##############################################################

				# If malware app, then uninstall the apk after every iteration and flash userdata/all partitions
				if(MALWARE_FLAG): 
					print("- Malware execution complete: Uninstall and check firmware.")
					# Uninstall the apk
					uninstall_apk(package_name)

					# Stop gnirehtet
					subprocess.Popen('gnirehtet stop', shell=True).wait()
					time.sleep(2)

					# Populate the checksum-after-malware field of all the partitions
					extract_partition_checksum(flag_before_malware= 0)

					# Check if non-userdata partitions have been modified
					modified_partition_list = check_partition_modification() 
					
					if modified_partition_list: # Checks if there are items in the list of modified partitions
						print("- System partitions modified: Flash entire firmware.")
						
						# Flash the entire firmware 
						firmware_flash_script.flash_all()

						# Log the details [datapoint : this malware modifies the following partition]
						modified_partition_names = ' '.join(modified_partition_list)
						
						log_string = "Apk name: "+apk_file+" | " + "Package name: "+package_name+" | "+ "Iteration: ("+str(ind_iter)+","+str(dep_iter+1)+")"+ " | "+"Seed: "+str(seed) +" | "+ "Malware: " +str(MALWARE_FLAG) + " | " + "Benign: " +str(BENIGN_FLAG) + " | " + "Modified partitions: " +str(modified_partition_names)						
						log_file_handler.error(log_string)

						# After firmware flashing is complete, populate the checksum-before-malware field of all the partitions
						extract_partition_checksum(flag_before_malware= 1)

						# Start gnirehtet again (Make sure that the relay server is started in a separate terminal)
						subprocess.Popen('gnirehtet start', shell=True).wait()
						time.sleep(2)

					else: 
						# System partitions have not been modified
						print("- Un-modified system partitions: Reboot device.")

						# Reboot the device
						reboot_device()
						# Wait for services to start up
						time.sleep(5)

						# Start gnirehtet again
						subprocess.Popen('gnirehtet start', shell=True).wait()
						time.sleep(2)

				# Uninstall the benign app after every iteration [It will be installed again at the beginning of the iteration]	
				elif(BENIGN_FLAG):
					print("- Benignware execution complete: Uninstall APK")

					# Stop gnirehtet
					subprocess.Popen('gnirehtet stop', shell=True).wait()
					time.sleep(2)
					
					# Uninstall the apk
					uninstall_apk(package_name)

					# Reboot the device
					reboot_device()
					# Wait for services to start up
					time.sleep(5)

					# Start gnirehtet again
					subprocess.Popen('gnirehtet start', shell=True).wait()
					time.sleep(2)

		# Data collected for the entire application. Push this folder to Dropbox
		app_folder_path =  dest_folder+'/'+package_dir_name
		dest_folder_path = '/' + dest_folder +'/'+package_dir_name # Need "/" in the beginning else dropbox throws error
		remove_folder = 1 # Flag for removing the folder. 1 means remove.
		file_flag = 0 # You are uploading a folder, not a file

		print(f"++++++ Pushing the application folder to dropbox: {app_folder_path} ++++++")

		# Start a new process for uploading to dropbox. In the meantime, continue collecting the data.
		upload_process = Process(target=push_to_dropbox, name="Uploader", args=(app_folder_path, dest_folder_path, remove_folder, file_flag, apk_file_path))
		upload_process.start()
		# Wait for the upload to finish
		upload_process.join()

		######################## Flash the data partition (Done after data-collection of every apk) ########################
		time.sleep(2)
		firmware_flash_script.flash_data_partition()

		if MALWARE_FLAG:
			# After firmware flashing is complete, populate the checksum-before-malware field of all the partitions
			extract_partition_checksum(flag_before_malware= 1)	
		
		# Start gnirehtet again 
		subprocess.Popen('gnirehtet start', shell=True).wait()
		time.sleep(2)
		####################################################################################################################
		
	# Stop gnirehtet
	subprocess.Popen('gnirehtet stop', shell=True, stdout=subprocess.PIPE).wait()	
	
	# Data collection for all the applications in path_dir is completed. Uploading the log file.
	# The dest folder contains only the log file now, since you delete the data for packages once you have uploaded it to dropbox (not true while testing)
	app_folder_path = dest_folder + '/log_file_' + app_type # Location of the log file
	dest_folder_path = '/' + dest_folder + '/log_file_' + app_type # Location of the log file in dropbox
	remove_folder = 0 # Flag for removing the folder. 1 means remove.
	file_flag = 1 # You are uploading a single file
	print(f"++++++++++++++++++++++++++++++++ Pushing the log file to dropbox: {app_folder_path} ++++++++++++++++++++++++++++++++")
	push_to_dropbox(app_folder_path, dest_folder_path, remove_folder, file_flag, apk_file_path) # Do not remove the log files from your system

def testing_timeout():
	time.sleep(2)

def main():

	# Dict storing the type of dataset, path of the directory storing the apks of the dataset, and flag for whether the directory contains malware
	app_dir = {
		
		## Benign samples for the std-dataset 
		"std_benign":['/home/harshit/research/androidzoo/usenix/std_benign/',0]

		## Malware samples for the std-dataset 
		# "std_malware":['/home/harshit/research/androidzoo/usenix/std_malware/',1]
		############################################################################################################################################


	}
			
	# Start collecting data for each of the directory
	for key, value in app_dir.items():		
		path_dir = value[0]
		MALWARE_FLAG = value[1]
		dest_folder = 'logs_'+key 
		os.system('mkdir -p ' + dest_folder)
		log_file_name = dest_folder+'/'+"log_file_"+key	
		# Setting up the log file
		log_file_handler = setup_logger(key, log_file_name, level=logging.WARNING)

		# Start data collection for this folder
		collect_data(path_dir= path_dir, 
					MALWARE_FLAG= MALWARE_FLAG, 
					dest_folder= dest_folder, 
					log_file_handler= log_file_handler,
					app_type= key)


if(__name__=="__main__"):
	main()
