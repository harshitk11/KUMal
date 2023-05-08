"""
Python script to train and evaluate all the first stage and the second stage models of XMD.
"""
import argparse
import datetime
# import stat
# import torch
from utils import Config
import os
import shutil
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode
import re
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from create_featEngineer_dataset import dataloader_generator
from prettytable import PrettyTable
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import MSWORD_FRIENDLY
from scipy.stats import entropy
from sklearn.metrics import roc_curve, auc
from extract_metainfo_from_parserinfo import get_dataset_metainfo


BENIGN_LABEL = 0
MALWARE_LABEL = 1

def get_args(kumal_base_folder_location):
    """
    Reads the config file and returns the config parameters.
    params:
        - kumal_base_folder_location: Location of xmd's base folder
    Output:

    """
    parser = argparse.ArgumentParser(description="KUMal: Concept Drift study of Hardware Performance Counter (HPC) based Malware Detectores")
    # Location of the default and the update config files.
    parser.add_argument('-config_default', type=str, dest='config_default', default=os.path.join(kumal_base_folder_location,'config','default_config.yaml'), help="File containing the default experimental parameters.")
    parser.add_argument('-config', type=str, dest='config', default=os.path.join(kumal_base_folder_location,'config','update_config.yaml'), help="File containing the experimental parameters to update.")
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

class ImagePlottingTools:
    """
    Contains all the methods for creating the plots.
    """
    def multiLinePlot(df, performanceMetric, plotTitle, saveLocation=None):
        """
        Draws a multi line plot using seaborn.
        params:
            - df : Dataframe {'truncatedDuration':truncatedDuration_tracker, 'logcatRuntimeThreshold': logcatRuntimeThreshold_tracker, 'performanceMetric':performanceMetric_tracker}
            - performanceMetric : Used for labeling the yaxis
            - plotTitle : Title of the plot
            - saveLocation: If passed, then will save the plot at this location.
        Output:
            - Saves the plot at saveLocation
        """
        # Convert to wide-form representation
        # df_wide = df.pivot(index='truncatedDuration', columns='logcatRuntimeThreshold', values='performanceMetric')
        df_wide = df.pivot(index='logcatRuntimeThreshold', columns='truncatedDuration', values='performanceMetric') # for perf vs. lrt plot
        
        sns.set_style("ticks")
        sns.set(font_scale=1.2)
        sns.set(rc={'figure.figsize':(10,7)})
        sns.lineplot(data=df_wide, markers=True, markersize=15)

        plt.rcParams["font.weight"] = "bold"
        plt.xticks(weight="bold")
        plt.yticks(weight="bold")
        plt.title(plotTitle, weight="bold")
        # plt.xlabel("Truncated Duration", weight="bold")
        plt.xlabel("Logcat Runtime Threshold", weight="bold")
        plt.ylabel(performanceMetric, weight="bold")
        # plt.legend(title="logcatRuntimeThreshold", prop={'weight': 'bold'})
        plt.legend(title="Truncated Duration", prop={'weight': 'bold'})
        # plt.ylim(0.75, 0.9)
        sns.despine(top=True, right=True)
        plt.tight_layout()
        if saveLocation:
            plt.savefig(saveLocation, dpi=300)   


class performanceMetricAggregator:
    @staticmethod
    def getAggregatePerformanceMetric(HPC_classifire_inst, performanceMetricType):
        """
        Returns a single performance metric from the HPC_classifier_instance based on the performanceMetricType.

        params:
            - HPC_classifire_inst (HPC_Classifier): Instance of the HPC_Classifier object with all the performance metrics updated.
            - performanceMetricType (dict): {
                                        "class_of_interest": "benign", "malware", or "summary",
                                        "splitType": "training" or "testing",
                                        "hpc-group-name": "hpc-group-1", "hpc-group-2", "hpc-group-3", "hpc-group-4", or "all"
                                        "performanceMetricName": 'f1', 'precision', or 'recall'
                                    }
                                                            
        Output:
            - performanceMetric (int)
        """
        classOfInterest = performanceMetricType["class_of_interest"]
        splitType = performanceMetricType["splitType"]
        performanceMetricName = performanceMetricType["performanceMetricName"]
        hpc_group_name = performanceMetricType["hpc-group-name"]
        
        performancMetricScoreList = []    
        
        if splitType == "training":
            # Only summary scores are available for in-distribution traning data
            if hpc_group_name == "all":
                for grpName in HPC_classifire_inst.hpcGroupNameList:
                    pScore = HPC_classifire_inst.stage1ClassifierPerformanceMetrics[splitType][grpName][performanceMetricName]
                    performancMetricScoreList.append(pScore)
            else:
                performancMetricScoreList.append(HPC_classifire_inst.stage1ClassifierPerformanceMetrics[splitType][hpc_group_name][performanceMetricName])
        
        elif splitType == "testing":
            if hpc_group_name == "all":
                for grpName in HPC_classifire_inst.hpcGroupNameList:
                    pScore = HPC_classifire_inst.stage1ClassifierPerformanceMetrics[splitType][grpName][classOfInterest][performanceMetricName]
                    performancMetricScoreList.append(pScore)
            else:
                performancMetricScoreList.append(HPC_classifire_inst.stage1ClassifierPerformanceMetrics[splitType][hpc_group_name][classOfInterest][performanceMetricName])
                
        # Get the mean of the performanceMetric
        aggregatePerformanceMetric = sum(performancMetricScoreList)/len(performancMetricScoreList)
        return aggregatePerformanceMetric
        
class resample_dataset:
    """
    Contains all the methods for resampling the datasets to achieve the desired malware percentage.
    """

    def __init__(self, malwarePercent) -> None:
        # % of malware in the resampled dataset. The resampling is done by oversampling the benign class.
        self.malwarePercent = malwarePercent
    
    def __call__(self, y):
        """
        Returns a dict containing the number of samples for the benign and the malware class
        """
        target_stats = Counter(y)
        Bo = int((1-self.malwarePercent)*target_stats[MALWARE_LABEL]/self.malwarePercent)
        Mo = target_stats[MALWARE_LABEL]
        resampled_stats = {MALWARE_LABEL:Mo, BENIGN_LABEL:Bo}
        return resampled_stats

    def generate_sampling_indices(self, X, y):
        """
        Generates indices of the samples from the original dataset that are used in the new dataset.
        params:
            - X : (Nsamples, Nfeature)
            - y : (Nsamples, )

        Output:
            - ros.sample_indices_ : Indices of the samples selected. ndarray of shape (n_new_samples,)
        """
        rmInst = resample_dataset(malwarePercent=self.malwarePercent)
        if self.malwarePercent == 0.1:
            ros = RandomOverSampler(random_state=42, sampling_strategy=rmInst)
        elif self.malwarePercent == 0.5:
            ros = RandomOverSampler(random_state=42, sampling_strategy='auto')
        else:
            raise ValueError(f"Expecting malwarePercent of 0.1 or 0.5. Got {self.malwarePercent}.")

        _, _ = ros.fit_resample(X, y)
        return ros.sample_indices_

    def resampleBaseTensor(self, X, y, filePathList):
        """
        Resamples the Hpc Tensor.
        params:
            - X: dataset (Nchannels, Nsamples, feature_size)
            - y: labels (Nsamples,) 
            - filePathList: (Nsamples,)
        Output:
            - X_res, y_res, filePathList_res : Resampled dataset
        """
        # Get the sampling indices
        sampIndx = self.generate_sampling_indices(X[0], y)

        # Resample the dataset
        X_res = X[:,sampIndx,:]
        y_res = y[sampIndx]
        filePathList_res = filePathList[sampIndx]

        return X_res, y_res, filePathList_res

    def resampleHpcTensor(self, Xlist, yList, filePathList):
        """
        Resamples the dataset for all the HPC groups.
        
        params:
            - Xlist: [dataset_group1, dataset_group2, dataset_group3, dataset_group4] | dataset shape: (1, Nsamples, feature_size)
            - yList: [labels_group1, labels_group2, labels_group3, labels_group4] | labels_shape: (Nsamples,)
            - filePathList: [filePath_group1, filePath_group2, filePath_group3, filePath_group4] | filePath_shape: (Nsamples,)

        Output:
            - Xlist_res, yList_res : Resampled dataset
        """
        Xlist_res = []
        yList_res = []
        filePathList_res = []

        for grpIndx, _ in enumerate(Xlist):
            X_res, y_res, filePath_res = self.resampleBaseTensor(X=Xlist[grpIndx], y=yList[grpIndx], filePathList=filePathList[grpIndx])
            Xlist_res.append(X_res)
            yList_res.append(y_res)
            filePathList_res.append(filePath_res)

        return Xlist_res, yList_res, filePathList_res

    @staticmethod
    def unitTestResampler():
        # Loading the datasets for testing the resampler
        hpc_x_test = []
        hpc_y_test = []
        hpc_filePaths_test = []
        for group in ["rn1","rn2","rn3","rn4"]:
            hpc_x_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset/std-dataset/0/10/{group}/channel_bins_train.npy"))
            hpc_y_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset/std-dataset/0/10/{group}/labels_train.npy"))
            with open(f"/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset/std-dataset/0/10/{group}/file_paths_train.npy", 'rb') as fp:
                hpc_filePaths_test.append(np.array(pickle.load(fp), dtype="object"))
            
        print(" [Pre] Details of HPC data")
        print(f" - Shape of the (hpc data, label, file_path) : {[(dataset.shape,labels.shape, filePaths.shape) for dataset,labels,filePaths in zip(hpc_x_test,hpc_y_test, hpc_filePaths_test)]}")
        print(" - [Pre] Number of malware and benign samples")
        print([Counter(Y) for Y in hpc_y_test])
        
        rmInst = resample_dataset(malwarePercent=0.5)
        hpc_x_test, hpc_y_test, hpc_filePaths_test = rmInst.resampleHpcTensor(Xlist=hpc_x_test, yList=hpc_y_test, filePathList=hpc_filePaths_test)
        
        print(" -------------------------------------------------------------------------------------------------")
        print(" [Post] Details of resampled HPC data")
        print(f" - Shape of the (hpc data, label, file_path) : {[(dataset.shape,labels.shape, filePaths.shape) for dataset,labels,filePaths in zip(hpc_x_test,hpc_y_test, hpc_filePaths_test)]}")
        print(" - Number of malware and benign samples")
        print([Counter(Y) for Y in hpc_y_test])
        

class baseRFmodel:
    """
    Object for the base Random Forest model. Tracks the different attributes of the classifiers.
    Contains all the methods for training, evaluation, saving, and loading the model.
    """
    
    def __init__(self, args, channelName=None) -> None:
        # Contains the parameters for training
        self.args = args
        # Channel for which the RF model is created
        self.channelName = channelName
        # Hyper parameter grid over which the tuning needs to be performed
        self.hyperparameterGrid = baseRFmodel.generate_hyperparameter_grid()
        
        ##################### Populated after training or loading a trained model ##################### 
        # List of validation scores for different hyperparameters
        self.validationScoreList = None
        # Stores the trained RF model
        self.trainedRFmodel = None
        # Stores the parameters of the best model
        self.bestModelParams = None
        ###############################################################################################
    
    def train(self, Xtrain, Ytrain):
        """
        Trains the Random Forest model. Tunes the trained model if modelTuneFlag is passed.
        
        params:
            - Xtrain: dataset (Nsamples, feature_size)
            - Ytrain: labels (Nsamples,) 
        """
        # Sanity check
        if len(Xtrain.shape)>2:
            raise ValueError(f"Shape of training data array incorrect : {Xtrain.shape}")

        rf_clf=RandomForestClassifier()
        
        rf_random = RandomizedSearchCV(estimator = rf_clf, 
                                        param_distributions = self.hyperparameterGrid, 
                                        n_iter = self.args.num_comb_model_tuning, 
                                        cv = self.args.num_cv, 
                                        verbose=1, 
                                        random_state=self.args.seed, 
                                        n_jobs = self.args.n_jobs,
                                        scoring='f1_macro')

        # Fit the random search model
        rf_random.fit(Xtrain, Ytrain)
        
        # Save the best model and its details
        self.bestModelParams = rf_random.best_params_
        self.trainedRFmodel = rf_random.best_estimator_
        self.validationScoreList = rf_random.cv_results_['mean_test_score'] 
        print(f"  - Validation score during training: {self.validationScoreList}")

    def eval(self, Xtest, Ytest = None, print_performance_metric = False):
        """
        Generates predictions from the trainedRFmodel. If Ytest is passed, then will generate accuracy metrics for the trained model.
        
        params:
            - Xtest: dataset (Nsamples, feature_size)
            - Ytest: labels (Nsamples,)

        Output:
            - predict_labels: Predicted labels from the trained model (Nsamples,)
            - test_performance_metric: Contains the performance metrics (f1, precision, recall) if the true labels are passed
        """
        test_performance_metric = None
        # Sanity check
        if len(Xtest.shape)>2:
            raise ValueError(f"Shape of testing data array incorrect : {Xtest.shape}")

        # Getting the prediction from the model over the test_data
        predict_labels = self.trainedRFmodel.predict(Xtest)

        if Ytest is not None:
            # Get the classification report of the prediction
            class_results = classification_report(y_true= Ytest,
                                                   y_pred= predict_labels, 
                                                   output_dict=True)
            
                        
            test_performance_metric = {"summary": {'f1': round(class_results['macro avg']['f1-score'], 2),
                                       'precision': round(class_results['macro avg']['precision'], 2),
                                       'recall': round(class_results['macro avg']['recall'], 2),
                                       'support': class_results['macro avg']['support']},
                                       
                           "class-1": {'f1': round(class_results['1']['f1-score'], 2),
                                        'precision': round(class_results['1']['precision'], 2),
                                        'recall': round(class_results['1']['recall'], 2),
                                        'support': class_results['1']['support']},
                                       
                           "class-0": {'f1': round(class_results['0']['f1-score'], 2),
                                        'precision': round(class_results['0']['precision'], 2),
                                        'recall': round(class_results['0']['recall'], 2),
                                        'support': class_results['0']['support']}}

            
            
            if print_performance_metric:
                # Print the classification report and the confusion matrix
                print(f" ----- Evaluation performation metrics -----")
                print(classification_report(Ytest,predict_labels))
                print(confusion_matrix(Ytest, predict_labels))

        return predict_labels, test_performance_metric 

    def eval_uncertainty(self, Xtest):
        """
        Generates an uncertainty score for each sample in Xtest.
                
        params:
            - Xtest: dataset (Nsamples, feature_size)
        
        Output:
            - uncertainty_score_np: Uncertainty score for each sample in Xtest (Nsamples,)
            - predict_proba: Probability of each class for each sample in Xtest (Nsamples,)
        """
        ensemble_of_classifiers = self.trainedRFmodel.estimators_
        num_classifiers = len(ensemble_of_classifiers)

        # Initialize an empty array to store the predictions from all classifiers
        prediction_array = np.zeros((len(Xtest), num_classifiers))

        # Iterate over each classifier in the ensemble
        for i, classifier in enumerate(ensemble_of_classifiers):
            # print(f"  - Evaluating classifier {i+1}/{num_classifiers}")
            predictions = classifier.predict(Xtest)
            prediction_array[:, i] = predictions

        # Get the probability of each class for each sample
        histogram_array = np.column_stack((np.sum(prediction_array == 0, axis=1), np.sum(prediction_array == 1, axis=1))) 
        uncertainty_score_array = entropy(histogram_array, axis=1, base=2)
        predict_proba = np.mean(prediction_array, axis=1)
            
        return uncertainty_score_array, predict_proba
         
    def save_model(self, fpath):
        """
        Saves the model and the model details to the specified path.

        params:
            - fpath: full path where the model should be saved
        """
        # Used for saving and loading the object
        model_pickle_file = {"channelName":self.channelName,
                                "validationScoreList":self.validationScoreList,
                                "trainedRFmodel":self.trainedRFmodel,  
                                "bestModelParams":self.bestModelParams}

        # Write the pickle file
        with open(fpath, 'wb') as handle:
            pickle.dump(model_pickle_file, handle, protocol=pickle.HIGHEST_PROTOCOL)    

    def load_model(self, fpath):
        """
        Loads the model from the specified path. And populates all the corresponding model details of this object.

        params:
            - fpath: full path from which to load the RF model
        """
        # Load the dict from fpath and update the instance attributes
        with open(fpath, 'rb') as handle:
            model_pickle_file = pickle.load(handle)

        self.channelName = model_pickle_file["channelName"]
        self.validationScoreList = model_pickle_file["validationScoreList"]
        self.trainedRFmodel = model_pickle_file["trainedRFmodel"]
        self.bestModelParams = model_pickle_file["bestModelParams"]


    @staticmethod
    def generate_hyperparameter_grid():
        """
        Generates the hyperparameter grid over which model tuning needs to be performed.
        
        Output:
            - Populates self.hyperparameterGrid
        """
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 80, stop = 200, num = 10)]
        # Criterion
        criterion = ["entropy","gini"]
        # Number of features to consider at every split
        max_features = ['sqrt', 'log2', None]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        randomGrid = {'n_estimators': n_estimators,
                    'criterion': criterion,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
        
        return randomGrid
    
    
        
    @staticmethod
    def unit_test_baseRFmodel(args):
        """
        unit test for the base RF model object
        """
        # Loading the datasets for testing the model
        hpc_x_train = np.load(f"/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset/std-dataset/10/10/rn1/channel_bins_train.npy")
        hpc_y_train = np.load(f"/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset/std-dataset/10/10/rn1/labels_train.npy")
        
        hpc_x_test = np.load(f"/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset/std-dataset/80/10/rn1/channel_bins_test.npy")
        hpc_y_test = np.load(f"/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset/std-dataset/80/10/rn1/labels_test.npy")
        
        print(f" - Shape of the training data and label : {hpc_x_train.shape, hpc_y_train.shape}")
        print(f" - Shape of the test data and label : {hpc_x_test.shape, hpc_y_test.shape}")
        
        # print(f" - Training the model -")
        # baseModelInst = baseRFmodel(args=args, channelName="test")
        # baseModelInst.train(Xtrain=hpc_x_train[0],Ytrain=hpc_y_train)

        # print(f" - Evaluating the model -")
        # _,test_performance_metric = baseModelInst.eval(Xtest=hpc_x_test[0],Ytest=hpc_y_test,print_performance_metric=True)
        # print(test_performance_metric)
        
        # print(f" - Saving the model -")
        # baseModelInst.save_model(fpath="testmodel.pkl")

        print(f" - Loading and testing the model -")
        newBaseModelInst = baseRFmodel(args=args)
        newBaseModelInst.load_model(fpath="testmodel.pkl")

        print(f" - Generate the uncertainty scores -")
        test_uncertainty, predict_proba = newBaseModelInst.eval_uncertainty(Xtest=hpc_x_test[0])
        print(np.mean(test_uncertainty))
        
        print(f" - Generate the ROC curve -")
        baseRFmodel.generate_roc_curve(Ytest=hpc_y_test, Ypred_proba=predict_proba, savePath="test2.png", fpr_threshold=0.2)
        # newBaseModelInst.eval(Xtest=X_test[0],Ytest=Y_test,print_performance_metric=True)

class HPC_classifier:
    """
    Object for tracking the performance of the HPC classifier for a given configuration of training and test data.
    """        
    hpcGroupNameList = ["hpc-group-1", "hpc-group-2", "hpc-group-3", "hpc-group-4"]

    def __init__(self, args) -> None:
        self.args = args
        # Stores a dict of baseRFmodel objects. One for every hpc group. {grpName: baseRFModel object, ...}
        self.hpcGroupBaseClf = None
        
        ########################################## Performance metricts  ##########################################
        # List of file hashes used for training the base classifiers
        self.baseClassifierTrainFileHashList = None 
        # List of file hashes used for testing the base classifiers
        self.baseClassifierTestFileHashList = None 
         
        # List of performance metrics for all the base classifiers of GLOBL channels and HPC groups. 
        stage1ClfPerformanceMetricTemplate = {chnGrpName: None for chnGrpName in HPC_classifier.hpcGroupNameList}
        self.stage1ClassifierPerformanceMetrics = {"training":stage1ClfPerformanceMetricTemplate.copy(), "testing":stage1ClfPerformanceMetricTemplate.copy()}
        ###########################################################################################################
    
    
    def stage1trainHPC(self, XtrainHPC, YtrainHPC, updateObjectPerformanceMetrics):
            """
            Trains all the base-classifiers for all the HPC groups.
            
            params:
                - XtrainHPC: [dataset_group1, dataset_group2, dataset_group3, dataset_group4] | dataset shape: (1, Nsamples, feature_size)
                - YtrainHPC: [labels_group1, labels_group2, labels_group3, labels_group4] | labels_shape: (Nsamples,)
                - updateObjectPerformanceMetrics: If True, then will update the performance metrics of the globl base classifiers.
                
            Output:
                - Populates the self.hpcGroupBaseClf
                - Updates the self.stage1ClassifierPerformanceMetrics if the flag is passed
            """
            # Dict of baseRFmodel objects
            modelDict = {}
            
            # Train a classifier for every group
            for groupNumber, groupName in enumerate(self.hpcGroupNameList): 
                print(f" - Training baseRF model for hpc-group: {groupName}")

                # Fetch trainingData (Nsamples, feature_size) and trainingLabel (Nsamples,)
                trainingData = XtrainHPC[groupNumber].squeeze() 
                trainingLabel = YtrainHPC[groupNumber]
                
                baseModelInst = baseRFmodel(args=self.args, channelName=groupName)
                baseModelInst.train(Xtrain=trainingData,Ytrain=trainingLabel)
                modelDict[groupName] = baseModelInst

                if updateObjectPerformanceMetrics:
                    _performance_metric = {'f1': max(baseModelInst.validationScoreList),
                                            'precision': None,
                                            'recall': None}
                    self.stage1ClassifierPerformanceMetrics["training"][groupName] = _performance_metric
                
            self.hpcGroupBaseClf = modelDict


    def stage1evalHPC(self, XtestHPC, YtestHPC, updateObjectPerformanceMetrics, print_performance_metric):
        """
        Evaluates all the base-classifiers for all the HPC groups.
        
        params:
            - XtestHPC: [dataset_group1, dataset_group2, dataset_group3, dataset_group4] | dataset shape: (1, Nsamples, feature_size)
            - YtestHPC: [labels_group1, labels_group2, labels_group3, labels_group4] | labels_shape: (Nsamples,)
            - updateObjectPerformanceMetrics: If True, then will update the performance metrics of the hpc-group base-classifiers.
            - print_performance_metric: If True, then will print the performance metrics to stdout

        Output:
            - allGroupPredictions: Predicted labels from the trained model 
                                [labels_group1, labels_group2, labels_group3, labels_group4] | labels-shape: (Nsamples,)
        """
        # Stores the predictions of all the groups over their corresponding test dataset
        allGroupPredictions = []
        
        # Test a classifier for every group
        for groupNumber, (groupName,groupModel) in enumerate(self.hpcGroupBaseClf.items()): 
            print(f" - Evaluating baseRF model for hpc-group: {groupName}")
            
            # Fetch testingData (Nsamples, feature_size) and trainingLabel (Nsamples,)
            testData = XtestHPC[groupNumber].squeeze() 
            testLabel = YtestHPC[groupNumber]
            
            # Get the prediction from the group model
            predict_labels, test_performance_metric = groupModel.eval(Xtest = testData, 
                                                                        Ytest = testLabel, 
                                                                        print_performance_metric = print_performance_metric)
            allGroupPredictions.append(predict_labels)
            
            if updateObjectPerformanceMetrics:
                self.stage1ClassifierPerformanceMetrics["testing"][groupName] = test_performance_metric
            
        return allGroupPredictions
        
    
    def pretty_print_performance_metric(self):
        """
        Prints the performance metric to stdout.
        """
        print("\n----------- HPC classifiers performance metric for all groups -----------")
        for splitType, stage1ClfPerformanceMetricTemplate in self.stage1ClassifierPerformanceMetrics.items():
            print(f"----------- Split Type : {splitType} -----------")
            if splitType == "testing":
                for chnGrpName, perfMetric in stage1ClfPerformanceMetricTemplate.items():
                    try:
                        print(f" Channel name: {chnGrpName}")
                        print(f" SUMMARY - F1 : {perfMetric['summary']['f1']} | precision : {perfMetric['summary']['precision']} | recall : {perfMetric['summary']['recall']} | support : {perfMetric['summary']['support']}")
                        print(f" MALWARE - F1 : {perfMetric['class-1']['f1']} | precision : {perfMetric['class-1']['precision']} | recall : {perfMetric['class-1']['recall']} | support : {perfMetric['class-1']['support']}")
                        print(f" BENIGN - F1 : {perfMetric['class-0']['f1']} | precision : {perfMetric['class-0']['precision']} | recall : {perfMetric['class-0']['recall']} | support : {perfMetric['class-0']['support']}")
                        
                    except:
                        print(f" *********** ")
            
            elif splitType == "training":
                for chnGrpName, perfMetric in stage1ClfPerformanceMetricTemplate.items():
                    print(f" Channel name: {chnGrpName}")
                    try:
                        print(f" SUMMARY - F1 : {perfMetric['f1']}")
                    except:
                        print(f" *********** ")
            
            else: raise ValueError(f"Invalid splitType: {splitType}")

       
    @staticmethod
    def get_train_test_hashList(file_paths_list_for_all_groups):
        """
        Returns a list of hashes (apks) used for training/testing the model for all the HPC-groups.
        params:
            - file_paths_list_for_all_groups (list): [file_paths_group1, file_paths_group2, file_paths_group3, file_paths_group4]
                                                    -> file_paths_group1 (list): List of file paths for group1 ...

        Output:
            - hashList (list): List of hashes extracted from the file paths
            
        """
        hashList = []
        regex_pattern = r'.*\/(.*)__.*it(\d*)_rn(\d*).txt'

        # Parse this file list to extract the hashes

        for file_pathList_for_group in file_paths_list_for_all_groups:
            for filename in file_pathList_for_group:
                file_hash_obj = re.search(regex_pattern, filename, re.M|re.I)
                file_hash_string = file_hash_obj.group(1).strip()
                if file_hash_string not in hashList:
                    hashList.append(file_hash_string)

        return hashList
    
    @staticmethod
    def get_metaInfo_from_fileList(file_paths_list):
        """
        For each path in file_paths_list, extracts the hash, it, rn
        params:
            - file_paths (list): List of file paths

        Output:
            - metaInfo (list): [(hash, it, rn), (hash, it, rn), ...)]
        """
        metaInfo = []
        regex_pattern = r'.*\/(.*)__.*it(\d*)_rn(\d*).txt'

        # Parse this file list to extract the hashes
        for filename in file_paths_list:
            file_hash_obj = re.search(regex_pattern, filename, re.M|re.I)
            
            file_hash_string = file_hash_obj.group(1).strip()
            iter_val = int(file_hash_obj.group(2).strip())
            rn_val = int(file_hash_obj.group(3).strip())
            
            metaInfo.append((file_hash_string, iter_val, rn_val))

        return metaInfo

    def save_HPC_clf_object(self, fpath):
        """
        Saves the model and the model details to the specified path.

        params:
            - fpath: full path where the model should be saved
        """
        # Used for saving and loading the object
        model_pickle_file = {   
                                "hpcGroupBaseClf":self.hpcGroupBaseClf,
                                "baseClassifierTrainFileHashList":self.baseClassifierTrainFileHashList,
                                "baseClassifierTestFileHashList":self.baseClassifierTestFileHashList,
                                "stage1ClassifierPerformanceMetrics":self.stage1ClassifierPerformanceMetrics,  
                            }

        # Write the pickle file
        with open(fpath, 'wb') as handle:
            pickle.dump(model_pickle_file, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        

    def load_HPC_clf_object(self, fpath):
        """
        Loads the model from the specified path. And populates all the corresponding model details of this object.
        """
        # Load the dict from fpath and update the instance attributes
        with open(fpath, 'rb') as handle:
            model_pickle_file = pickle.load(handle)

        self.hpcGroupBaseClf = model_pickle_file["hpcGroupBaseClf"]
        self.baseClassifierTrainFileHashList = model_pickle_file["baseClassifierTrainFileHashList"]
        self.baseClassifierTestFileHashList = model_pickle_file["baseClassifierTestFileHashList"]
        self.stage1ClassifierPerformanceMetrics = model_pickle_file["stage1ClassifierPerformanceMetrics"]
        
    def generate_roc_curve_for_all_groups(self, XtrainHPC, YtrainHPC, savePath=None):
        """
        Generates the roc curve for all the groups.
        Flow:
            - Generate the predict_proba for all the groups
            - Generate the roc curve for all the groups
        params:
            - XtrainHPC (list): List of training data for all the groups [group1, group2, group3, group4] | shape: (Nsamples, feature_size)
            - YtrainHPC (list): List of training labels for all the groups [group1, group2, group3, group4] | shape: (Nsamples,)
            - savePath (str): Path where the roc curve should be saved
        """
        # Stores the predictions of all the groups over their corresponding test dataset
        allGroup_PredictionProba = []
        
        # Test a classifier for every group
        for groupNumber, (groupName,groupModel) in enumerate(self.hpcGroupBaseClf.items()): 
            print(f" - Evaluating baseRF model for hpc-group: {groupName}")
            
            # Fetch testingData (Nsamples, feature_size) and trainingLabel (Nsamples,)
            testData = XtrainHPC[groupNumber].squeeze() 
            testLabel = YtrainHPC[groupNumber]
            
            # Get the prediction from the group model
            _, predict_proba = groupModel.eval_uncertainty(Xtest = testData)
            allGroup_PredictionProba.append(predict_proba)
            
        linestyle = ['solid', 'dashed', 'dashdot', ':']
        linecolor = ['#4c72b0', '#55a868', '#c44e52', '#8172b2']

        # Generate the roc curve for all the groups
        for i in range(len(YtrainHPC)):
            fpr, tpr, thresholds = roc_curve(YtrainHPC[i], allGroup_PredictionProba[i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2, label=f'HPC-group-{i+1} (AUC = {roc_auc:.2f})', linestyle=linestyle[i], color=linecolor[i])

        plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=12)
        plt.tick_params(labelsize=10, width=2, length=5, which='both')
        plt.grid(True)
        plt.tight_layout()  # Optional for better spacing between elements

        if savePath:
            plt.savefig(savePath, dpi=300)  # Save the figure if needed
        plt.close()
        
        return allGroup_PredictionProba
    
    def generate_uncertainty_score_for_all_groups(self, XtestHPC):
        """
        Generates the uncertainty score for all the groups.
        """
        pass

    @staticmethod
    def unit_test_HPC_classifier(args):
        """
        unit test for the late_stage_fusion object
        """
        ############################### Testing the base classifier modules ###############################
        # Loading the unmatched datasets for the unit tests
        hpc_x_train = []
        hpc_y_train = []
        hpc_filePaths_train = []
        hpc_x_test = []
        hpc_y_test = []
        hpc_filePaths_test = []
        
        for group in ["rn1","rn2","rn3","rn4"]:
            hpc_x_train.append(np.load(f"/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset/std-dataset/0/10/{group}/channel_bins_train.npy"))
            hpc_y_train.append(np.load(f"/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset/std-dataset/0/10/{group}/labels_train.npy"))
            with open(f"/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset/std-dataset/0/10/{group}/file_paths_train.npy", 'rb') as fp:
                hpc_filePaths_train.append(np.array(pickle.load(fp), dtype="object"))
               
            # hpc_x_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset/cdyear1-dataset/0/10/{group}/channel_bins_test.npy"))
            # hpc_y_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset/cdyear1-dataset/0/10/{group}/labels_test.npy"))

            hpc_x_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset/std-dataset/0/10/{group}/channel_bins_test.npy"))
            hpc_y_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset/std-dataset/0/10/{group}/labels_test.npy"))
            with open(f"/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset/std-dataset/0/10/{group}/file_paths_test.npy", 'rb') as fp:
                hpc_filePaths_test.append(np.array(pickle.load(fp), dtype="object"))
            

        print("Details of HPC training and test data")
        print(f" - Shape of the training data and label : {[(dataset.shape,labels.shape) for dataset,labels in zip(hpc_x_train,hpc_y_train)]}")
        print(f" - Shape of the test data and label : {[(dataset.shape,labels.shape) for dataset,labels in zip(hpc_x_test,hpc_y_test)]}")
        
        #################### Resampler ####################
        rmInst = resample_dataset(malwarePercent=0.5)
        hpc_x_train, hpc_y_train, hpc_filePaths_train = rmInst.resampleHpcTensor(Xlist=hpc_x_train, yList=hpc_y_train, filePathsList=hpc_filePaths_train)
        hpc_x_test, hpc_y_test, hpc_filePaths_test= rmInst.resampleHpcTensor(Xlist=hpc_x_test, yList=hpc_y_test, filePathsList=hpc_filePaths_test)
        ###################################################
        
        # Testing the training module
        print(f" - Training the hpc and globl base classifiers -")
        HPC_clfInst = HPC_classifier(args=args)
        # HPC_clfInst.stage1trainHPC(XtrainHPC=hpc_x_train, YtrainHPC=hpc_y_train, updateObjectPerformanceMetrics=True)
        
        # print(" - Summary performance metric of all the models [post training] -")
        # HPC_clfInst.pretty_print_performance_metric()

        # # Testing the evaluation module
        # print(f" - Evaluating the hpc and globl base classifiers -")
        # allGroupPredictions = HPC_clfInst.stage1evalHPC(XtestHPC=hpc_x_test, YtestHPC=hpc_y_test, updateObjectPerformanceMetrics=True, print_performance_metric=True)

        # print(" - Shape of the predicted labels post evaluation -")
        # print(f" - HPC predicted labels for all groups (Nsamples, ): {[grp.shape for grp in allGroupPredictions]}")

        # print(" - Summary performance metric of all the models [post evaluation] -")
        # HPC_clfInst.pretty_print_performance_metric()
        
        
        # # Testing the loading and saving module
        # print(f" - Saving the object -")
        # HPC_clfInst.save_HPC_clf_object(fpath="testmodel.pkl")
        
        # print(f" - Loading the object and testing the models -")
        # HPC_clfInst_loaded = HPC_classifier(args=args)
        # HPC_clfInst_loaded.load_HPC_clf_object(fpath="testmodel.pkl")
        # HPC_clfInst_loaded.stage1evalHPC(XtestHPC=hpc_x_test, YtestHPC=hpc_y_test, updateObjectPerformanceMetrics=True, print_performance_metric=True)

        # print(" - Summary performance metric of all the models [post evaluation post loading] -")
        # HPC_clfInst_loaded.pretty_print_performance_metric()
        
        # print(" - Plot the roc curve for the loaded model -")
        # HPC_clfInst_loaded.generate_roc_curve_for_all_groups(XtrainHPC=hpc_x_test, YtrainHPC=hpc_y_test, savePath="test_roc.png")
        
        # Testing the hashList extraction module
        print(f" - Extracting the hashList -")
        hashList_train = HPC_clfInst.get_train_test_hashList(file_paths_list_for_all_groups=hpc_filePaths_train)
        hashList_test = HPC_clfInst.get_train_test_hashList(file_paths_list_for_all_groups=hpc_filePaths_test)
        metaInfo_train = HPC_clfInst.get_metaInfo_from_fileList(file_paths_list=hpc_filePaths_train[0])
        

class featureEngineeredDatasetLoader:
    """
    Contains all the helper methods used for loading the feature engineered dataset.
    """
    def __init__(self, basePath_featureEngineeredDataset, datasetName, logcatRuntimeThreshold, truncatedDuration):
        """
        Loads the dataset for all classification tasks of interest.

        params:
            - basePath_featureEngineeredDataset (str): Location of the base folder where all the feature engineered dataset is stored
            - datasetName (str): Can be one of the following {'std-dataset', 'cdyear1-dataset', 'cdyear2-dataset', 'cdyear3-dataset'}
            - logcatRuntimeThreshold (int), truncatedDuration (int) : Used for accessing the corresponding dataset
        
        Output:
            - featEngineeredData (dict): {"train": [X_list, y_list, file_list] , "test": [X_list, y_list, file_list]} 
                                            -> X_list (list): [X_group1, X_group2, ..., X_group4]
                                            -> y_list (list): [y_group1, y_group2, ..., y_group4]
                                            -> file_list (list): [file_group1, file_group2, ..., file_group4]
        """
        # Used for accessing the dataset
        self.basePath_featureEngineeredDataset = basePath_featureEngineeredDataset
        self.logcatRuntimeThreshold = logcatRuntimeThreshold
        self.truncatedDuration = truncatedDuration
        self.datasetName = datasetName

        # Partitions to be loaded
        datasetType, _ = dataloader_generator.get_dataset_type_and_partition_dist(dataset_name = datasetName)
        self.requiredPartitions = dataloader_generator.partition_activation_flags[datasetType]
        

    def load_dataset(self):
        """
        Loads the datasets for the different classification tasks of interest.
        Output:
            - featEngineeredData (dict): {"train": [X_list, y_list, file_list] , "test": [X_list, y_list, file_list]} 
                                            -> X_list (list): [X_group1, X_group2, ..., X_group4]
                                            -> y_list (list): [y_group1, y_group2, ..., y_group4]
                                            -> file_list (list): [file_group1, file_group2, ..., file_group4]
        """
        featureEngineeredData = {}

        for partition, activationFlag in self.requiredPartitions.items():
            if activationFlag:
                featureEngineeredData[partition] = self.load_rnBucket_dataset(partition_type=partition)
            else:
                featureEngineeredData[partition] = None
            
        return featureEngineeredData
                    
    def load_rnBucket_dataset(self, partition_type):
        """
        Used for loading partitions that are divided into rn buckets.

        params:
            - partition_type: "train" or "test"

        Output: X_list, y_list, file_list
            - X_list (ndarray): [dataset_group1, dataset_group2, dataset_group3, dataset_group4] | dataset shape: (1, Nsamples, feature_size)
            - y_list (ndarray): [labels_group1, labels_group2, labels_group3, labels_group4] | labels_shape: (Nsamples,)
            - file_list (ndarray): [files_group1, files_group2, files_group3, files_group4] | files_shape: (Nsamples,)
        """
        X_list, y_list, file_list = [], [], []
        for group in ["rn1","rn2","rn3","rn4"]:
            X_list.append(np.load(os.path.join(self.basePath_featureEngineeredDataset, self.datasetName, str(self.logcatRuntimeThreshold), str(self.truncatedDuration), group, f"channel_bins_{partition_type}.npy")))
            y_list.append(np.load(os.path.join(self.basePath_featureEngineeredDataset, self.datasetName, str(self.logcatRuntimeThreshold), str(self.truncatedDuration), group, f"labels_{partition_type}.npy")))
            with open(os.path.join(self.basePath_featureEngineeredDataset, self.datasetName, str(self.logcatRuntimeThreshold), str(self.truncatedDuration), group, f"file_paths_{partition_type}.npy"), 'rb') as fp:
                file_list.append(np.array(pickle.load(fp), dtype="object"))
            
        return X_list, y_list, file_list

    
    @staticmethod
    def unit_test_featureEngineeredDatasetLoader():
        loadDatasetInst = featureEngineeredDatasetLoader(basePath_featureEngineeredDataset="/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset",
                                                        datasetName="std-dataset",
                                                        logcatRuntimeThreshold=80,
                                                        truncatedDuration=10)
        featureEngineeredData = loadDatasetInst.load_dataset()
    
        for partitionName, partition in featureEngineeredData.items():
            if partition is None:
                print(f"Partition- {partitionName} | *********** ")
                continue
            print(f"Partition- {partitionName} | Size: {[(x.shape,y.shape,z.shape) for x,y,z in zip(partition[0],partition[1],partition[2])]}")
                

class orchestrator:
    """
    Orchestrates the training and evaluation tasks for all the datasets and classification tasks of interest.
    """
    
    def __init__(self, args, basePath_featureEngineeredDataset, datasetName, malwarePercent, kumal_base_folder_location) -> None:
        self.args = args
        # Location of the base folder where all the feature engineered datasets are stored
        self.basePath_featureEngineeredDataset = basePath_featureEngineeredDataset
        
        # List of logcat-runtime-thresholds and candidate-truncated-durations [Used for accessing the datasets]
        self.candidateLogcatRuntimeThresholds = [i for i in range(0, args.collected_duration, args.step_size_logcat_runtimeThreshold)]
        self.candidateTruncatedDurations = [i for i in range(args.step_size_truncated_duration, args.collected_duration+args.step_size_truncated_duration, args.step_size_truncated_duration)]

        # Name of the dataset 
        self.datasetName = datasetName

        # Percentage of malware in the dataset. Can take value 0.1 (for 10%) or 0.5 (for 50%)
        self.malwarePercent = malwarePercent

        # Base folder of xmd's dataset
        self.kumal_base_folder_location = kumal_base_folder_location


    def std_dataset_tasks(self, logcatRuntimeThreshold, truncatedDuration, print_performance_metric, saveTrainedModels, trainHPCClassifiers):
        """
        Tasks:
            -Load the dataset for all the classification tasks.
            -Resample the dataset.
            -Train the HPC base classifiers.
            -Save the hash list used for training the base classifiers
            -Save the trained base classifiers.
        params:
            - logcatRuntimeThreshold (int): Filter runtime threshold used for filtering the dataset.
            - truncatedDuration (int): To truncate the time series, i.e., take the first truncatedDuration seconds.
            - print_performance_metric (bool): If True, then print the performance metrics for all evaluations.
            - saveTrainedModels (bool): If True, then save the lateStageFusion object in the trainedModels folder.
            - trainHPCClassifiers (bool): If True, then train the HPC base classifiers.
        Output:
            - HPC_classifier_inst : Instance of the object storing all the trained models
        """
        ######################################### Load the dataset for all the tasks ########################################
        assert self.datasetName == "std-dataset", "Incorrect dataset name"
        loadDatasetInst = featureEngineeredDatasetLoader(basePath_featureEngineeredDataset=self.basePath_featureEngineeredDataset,
                                                        datasetName=self.datasetName,
                                                        logcatRuntimeThreshold=logcatRuntimeThreshold,
                                                        truncatedDuration=truncatedDuration)
        featureEngineeredData = loadDatasetInst.load_dataset()

        # For training the hpc base classifiers
        hpc_X_train, hpc_Y_train, hpc_fileList_train = featureEngineeredData['train']
        hpc_X_test, hpc_Y_test, hpc_fileList_test = featureEngineeredData['test']
        ######################################################################################################################
        
        ######################################## Resampler ########################################
        rmInst = resample_dataset(malwarePercent=self.malwarePercent)
        hpc_X_train, hpc_Y_train, hpc_fileList_train = rmInst.resampleHpcTensor(Xlist=hpc_X_train, yList=hpc_Y_train, filePathList=hpc_fileList_train)
        hpc_X_test, hpc_Y_test, hpc_fileList_test = rmInst.resampleHpcTensor(Xlist=hpc_X_test, yList=hpc_Y_test, filePathList=hpc_fileList_test)
        ###########################################################################################
        
        HPC_classifier_inst = None
        if trainHPCClassifiers:
            ######################################### Train the HPC classifiers ########################################
            print(f" - Training the hpc and globl base classifiers -")
            HPC_classifier_inst = HPC_classifier(args=self.args)
            HPC_classifier_inst.stage1trainHPC(XtrainHPC=hpc_X_train, YtrainHPC=hpc_Y_train, updateObjectPerformanceMetrics=True)
            ############################################################################################################
            
            ######################################### Test the HPC ########################################
            HPC_classifier_inst.stage1evalHPC(XtestHPC=hpc_X_test, 
                                              YtestHPC=hpc_Y_test, 
                                              updateObjectPerformanceMetrics=True, 
                                              print_performance_metric=True)
            ###############################################################################################
            
            # Save the hash list used for training and testing the base classifiers
            HPC_classifier_inst.baseClassifierTrainFileHashList = HPC_classifier.get_train_test_hashList(file_paths_list_for_all_groups=hpc_fileList_train)
            HPC_classifier_inst.baseClassifierTestFileHashList = HPC_classifier.get_train_test_hashList(file_paths_list_for_all_groups=hpc_fileList_test)
            
            if print_performance_metric:
                # Pretty print performance metric
                print(f"Summary of performance metrics for dataset : {self.datasetName}, logcatRuntimeThreshold : {logcatRuntimeThreshold}, truncatedDuration : {truncatedDuration}")
                HPC_classifier_inst.pretty_print_performance_metric()
        
            # Save the trained models
            if saveTrainedModels:
                savePath = os.path.join(self.kumal_base_folder_location, "res", "trainedModels", self.datasetName)
                if not os.path.isdir(savePath):
                    os.makedirs(savePath)
                savePath = os.path.join(savePath,f"HPC_Object_logRuntime{logcatRuntimeThreshold}_truncDuration{truncatedDuration}_malwarePercent{self.malwarePercent}.pkl")
                HPC_classifier_inst.save_HPC_clf_object(fpath=savePath)
        
        else:
            # Load the saved object containing the trained models
            savePath = os.path.join(self.kumal_base_folder_location, "res", "trainedModels", self.datasetName)
            savePath = os.path.join(savePath,f"HPC_Object_logRuntime{logcatRuntimeThreshold}_truncDuration{truncatedDuration}_malwarePercent{self.malwarePercent}.pkl")
            HPC_classifier_inst = HPC_classifier(args=self.args)
            HPC_classifier_inst.load_HPC_clf_object(fpath=savePath)

        return HPC_classifier_inst

    def cd_dataset_tasks(self, trainedModelDetails, logcatRuntimeThreshold, truncatedDuration, print_performance_metric, save_HPC_Clf_Object):
        """
        Tasks:
            -Load the trained model.
            -Load the dataset for all the classification tasks.
            -Resample the dataset.
            -Test HPC base classifier
            -Save the hash list used for testing the classifiers

        params:
            - trainedModelDetails (dict) : {"logcatRuntimeThreshold": (int), "truncatedDuration": (int), "malwarePercent": (float)} 
                                        -> Used for loading the corresponding trained model
            - logcatRuntimeThreshold (int), truncatedDuration (int): Used for loading the dataset for the classification task
            - print_performance_metric (bool): If True, then print the performance metrics for all evaluations.
            - save_HPC_Clf_Object (bool): If True, then save the updated HPC_classifier object.
        
        Output:
            - lateFusionInstance : Instance of the object storing all the update performance evaluation metrics        
        """
        ################################### Load the trained model ###################################
        savePath = os.path.join(self.kumal_base_folder_location, "res", "trainedModels", "std-dataset")
        savePath = os.path.join(savePath,f"HPC_Object_logRuntime{trainedModelDetails['logcatRuntimeThreshold']}_truncDuration{trainedModelDetails['truncatedDuration']}_malwarePercent{trainedModelDetails['malwarePercent']}.pkl")
        
        HPC_classifier_inst = HPC_classifier(args=self.args)
        HPC_classifier_inst.load_HPC_clf_object(fpath=savePath)
        ##############################################################################################

        ################################### Load the dataset for all classification tasks ###################################
        assert (self.datasetName == "cd-dataset") or (self.datasetName == "cdyear1-dataset") or (self.datasetName == "cdyear2-dataset") or (self.datasetName == "cdyear3-dataset"), "Incorrect dataset name"
        loadDatasetInst = featureEngineeredDatasetLoader(basePath_featureEngineeredDataset=self.basePath_featureEngineeredDataset,
                                                        datasetName=self.datasetName,
                                                        logcatRuntimeThreshold=logcatRuntimeThreshold,
                                                        truncatedDuration=truncatedDuration)
        featureEngineeredData = loadDatasetInst.load_dataset()
        hpc_X_test, hpc_Y_test, hpc_fileList_test = featureEngineeredData['test']
        #####################################################################################################################

        ######################################## Resampler ########################################
        rmInst = resample_dataset(malwarePercent=self.malwarePercent)
        hpc_X_test, hpc_Y_test, hpc_fileList_test = rmInst.resampleHpcTensor(Xlist=hpc_X_test, yList=hpc_Y_test, filePathList=hpc_fileList_test)
        ###########################################################################################

        ################################### Test the HPC and DVFS base classifiers ###################################
        print(f" - Testing the hpc and globl base classifiers -")
        HPC_classifier_inst.stage1evalHPC(XtestHPC=hpc_X_test, YtestHPC=hpc_Y_test, updateObjectPerformanceMetrics=True, print_performance_metric=print_performance_metric)
        ##############################################################################################################

        # Save the hash list used for testing 
        HPC_classifier_inst.baseClassifierTestFileHashList = HPC_classifier.get_train_test_hashList(file_paths_list_for_all_groups=hpc_fileList_test)
            
        if print_performance_metric:
            print(f"Summary of performance metrics for dataset : {self.datasetName}, logcatRuntimeThreshold : {logcatRuntimeThreshold}, truncatedDuration : {truncatedDuration}")
            HPC_classifier_inst.pretty_print_performance_metric()
        
        # Save the updated HPC objects
        if save_HPC_Clf_Object:
            savePath = os.path.join(self.kumal_base_folder_location, "res", "trainedModels", self.datasetName)
            if not os.path.isdir(savePath):
                os.mkdir(savePath)
            savePath = os.path.join(savePath,f"HPC_Object_logRuntime{logcatRuntimeThreshold}_truncDuration{truncatedDuration}_malwarePercent{self.malwarePercent}_trainedModel{trainedModelDetails['logcatRuntimeThreshold']}_{trainedModelDetails['truncatedDuration']}_{trainedModelDetails['malwarePercent']}.pkl")
            HPC_classifier_inst.save_HPC_clf_object(fpath=savePath)

        return HPC_classifier_inst

    

    def logcat_runtime_vs_truncated_duration_grid(self, trainHPCClassifiers, trainedModelDetails = None):
        """
        Performs a grid search over logcatRuntimeThreshold and truncatedDuration for the following tasks.
            Tasks:
                - For cd-dataset, generate evaluation scores using one of the following: 
                                (1) specified trained model [only when trainedModelDetails is passed], 
                                (2) trained model with the same config parameters: logcatRuntimeThreshold and truncatedDuration.
                - For std-dataset, generate late-stage-fusion instances storing the trained models.
            
            params:
                - trainHPCClassifiers (bool) : For std-dataset, if True, then will train the corresponding classifiers.
                - trainedModelDetails (dict) : {"logcatRuntimeThreshold": (int), "truncatedDuration" : (int), "malwarePercent": (float)}
                                                If this parameter is passed then grid search using the cd-dataset is performed using this trained model.
                                                Else, the same config trained model is used for testing the corresponding cd-dataset instance.
        """
        datasetType, _ = dataloader_generator.get_dataset_type_and_partition_dist(dataset_name = self.datasetName)
        
        if (datasetType == "cd-dataset"):
            # Grid search all parameter using the trained model.
            for logcatRuntimeThreshold in self.candidateLogcatRuntimeThresholds:
                for truncatedDuration in self.candidateTruncatedDurations:
                    print(f" ---------- Generating late-stage-fusion instance for logcatRuntimeThreshold {logcatRuntimeThreshold} and truncatedDuration {truncatedDuration} ----------")
                    # Trained model to be used for testing
                    orchInst = orchestrator(args=self.args, 
                                basePath_featureEngineeredDataset=self.basePath_featureEngineeredDataset, 
                                datasetName=self.datasetName, 
                                malwarePercent=self.malwarePercent,
                                xmd_base_folder_location=self.xmd_base_folder_location)
        
                    if trainedModelDetails is None:
                        # Use the trained model with the same config
                        savedTrainedModelDetails = {"logcatRuntimeThreshold": logcatRuntimeThreshold, "truncatedDuration" : truncatedDuration, "malwarePercent":self.malwarePercent}
                    else:
                        # Use the specified trained model
                        savedTrainedModelDetails = trainedModelDetails
 
                    orchInst.cd_dataset_tasks(trainedModelDetails=savedTrainedModelDetails,
                                            logcatRuntimeThreshold=logcatRuntimeThreshold, 
                                            truncatedDuration=truncatedDuration, 
                                            print_performance_metric=True, 
                                            save_HPC_Clf_Object=True)                
                    
        elif (datasetType == "std-dataset"):
            # Generate the trained models by doing gridsearch over logcatRuntimeThreshold and truncatedDuration.
            for logcatRuntimeThreshold in self.candidateLogcatRuntimeThresholds:
                for truncatedDuration in self.candidateTruncatedDurations:
                    orchInst = orchestrator(args=self.args, 
                                basePath_featureEngineeredDataset=self.basePath_featureEngineeredDataset, 
                                datasetName=self.datasetName, 
                                malwarePercent=self.malwarePercent,
                                kumal_base_folder_location=self.kumal_base_folder_location)
                    
                    orchInst.std_dataset_tasks(logcatRuntimeThreshold=logcatRuntimeThreshold, 
                                            truncatedDuration=truncatedDuration, 
                                            print_performance_metric=True, 
                                            saveTrainedModels=True,
                                            trainHPCClassifiers=trainHPCClassifiers)                
        
        else:
            raise ValueError(f"Incomplete arguments : datasetType is {datasetType} and trainedModelDetails is {trainedModelDetails}.")


    def prettyPrintGridPerformanceMetrics(self, datasetName, performanceMetricType, malwarePercent, trainedModelDetails=None):
        """
        Pretty prints the performance metrics over the search grid of logcatRuntimeThreshold and truncatedDuration.
        Edit this method if you want a different performance metric.
        
        params:
            - datasetName : Name of the dataset for which the performance metric grid needs to be printed [If dataset is cd, then check trainedModelDetails]
            - performanceMetricType (dict): {
                                                "class_of_interest": "benign", "malware", or "summary",
                                                "splitType": "training" or "testing",
                                                "hpc-group-name": "hpc-group-1", "hpc-group-2", "hpc-group-3", "hpc-group-4", or "all"
                                                "performanceMetricName": 'f1', 'precision', or 'recall'
                                            }
                                                                            
            - malwarePercent (int) : Used for accessing the corresponding HPC_classifier object
            - trainedModelDetails (dict) : {"logcatRuntimeThreshold": (int), "truncatedDuration": (int), "malwarePercent": (float)} 
                                        -> Used for loading the corresponding trained model
                                        -> If dataset is cd, and trainedModelDetails is specified, then the corresponding trained model
                                            is used for generating the grid.

        Output:
            - gridTable (PrettyTable) : Contains the performance metric over the grid of truncatedDuration and logcatRuntimeThreshold
                                Format: []
        """
        # Storing the grid as a df for plotting
        truncatedDuration_tracker = []
        logcatRuntimeThreshold_tracker = []
        performanceMetric_tracker = []
        
        datasetType, _ = dataloader_generator.get_dataset_type_and_partition_dist(dataset_name = datasetName)

        # For display
        gridTable = PrettyTable()
        gridTable.field_names = ["LogcatRuntimeThreshold"]+[truncatedDuration for truncatedDuration in self.candidateTruncatedDurations]

        for logcatRuntimeThreshold in self.candidateLogcatRuntimeThresholds:
            performanceMetricForAll_truncatedDuration = []

            for truncatedDuration in self.candidateTruncatedDurations:
                ################################### Load the saved HPC_classifier_instance ###################################
                savePath = os.path.join(self.kumal_base_folder_location, "res", "trainedModels", datasetName)
                
                if datasetType == "std-dataset":
                    savePath = os.path.join(savePath,f"HPC_Object_logRuntime{logcatRuntimeThreshold}_truncDuration{truncatedDuration}_malwarePercent{malwarePercent}.pkl")
                elif datasetType == "cd-dataset" and (trainedModelDetails is not None): # Testing wrt the specified trained model
                    savePath = os.path.join(savePath,f"HPC_Object_logRuntime{logcatRuntimeThreshold}_truncDuration{truncatedDuration}_malwarePercent{malwarePercent}_trainedModel{trainedModelDetails['logcatRuntimeThreshold']}_{trainedModelDetails['truncatedDuration']}_{trainedModelDetails['malwarePercent']}.pkl")
                elif datasetType == "cd-dataset": # Testing with the trained model of same config 
                    # savePath = os.path.join(savePath,f"HPC_Object_logRuntime{logcatRuntimeThreshold}_truncDuration{truncatedDuration}_malwarePercent{malwarePercent}_trainedModel{logcatRuntimeThreshold}_{truncatedDuration}_{malwarePercent}.pkl")
                    savePath = os.path.join(savePath,f"HPC_Object_logRuntime{logcatRuntimeThreshold}_truncDuration{truncatedDuration}_malwarePercent{malwarePercent}_trainedModel{0}_{truncatedDuration}_{malwarePercent}.pkl") #for perf vs lrt plot
                else: 
                    raise ValueError(f"Incorrect dataset name specified: {datasetName}") 

                HPC_classifier_inst = HPC_classifier(args=self.args)
                HPC_classifier_inst.load_HPC_clf_object(fpath=savePath)
                ##############################################################################################################

                #################################### Get the performance metric from the lateFusionInstance ###################################                                                                                        
                pScore = performanceMetricAggregator.getAggregatePerformanceMetric(HPC_classifire_inst=HPC_classifier_inst, 
                                                                            performanceMetricType=performanceMetricType)
                pScore = round(pScore,2)
                performanceMetricForAll_truncatedDuration.append(pScore)

                # For the plots
                truncatedDuration_tracker.append(truncatedDuration)
                logcatRuntimeThreshold_tracker.append(logcatRuntimeThreshold)
                performanceMetric_tracker.append(pScore)
                ###############################################################################################################################

            gridTable.add_row([logcatRuntimeThreshold]+performanceMetricForAll_truncatedDuration)
            
        print(f" --- Performance metric for {datasetName} --- \nperformanceMetricName: {performanceMetricType['performanceMetricName']}\
            \nhpc_group_name: {performanceMetricType['hpc-group-name']}\nClass-Of-Interest: {performanceMetricType['class_of_interest']}\
            \nSplitType: {performanceMetricType['splitType']}")
        
        print(gridTable)
        print("--------------------------------------------------")
        print(gridTable.get_csv_string())

        
        ####################################################### Generate the plot #######################################################
        d = {'truncatedDuration':truncatedDuration_tracker, 'logcatRuntimeThreshold': logcatRuntimeThreshold_tracker, 'performanceMetric':performanceMetric_tracker}
        df = pd.DataFrame(data=d)

        if trainedModelDetails is not None:
            trainedModelFileName = f"trainedModel{trainedModelDetails['logcatRuntimeThreshold']}_{trainedModelDetails['truncatedDuration']}_{trainedModelDetails['malwarePercent']}"
        else:
            trainedModelFileName = None

        ImagePlottingTools.multiLinePlot(df=df, 
                                        performanceMetric=performanceMetricType['performanceMetricName'], 
                                        plotTitle=f"Performance over lRT vs. tD grid {performanceMetricType['performanceMetricName']}",
                                        saveLocation=f"/data/hkumar64/projects/arm-telemetry/KUMal/plots/analysis/{datasetName}_{performanceMetricType['performanceMetricName']}_{performanceMetricType['class_of_interest']}_{performanceMetricType['hpc-group-name']}_{performanceMetricType['splitType']}_model_{trainedModelFileName}.png")
        #################################################################################################################################

        return gridTable

    @staticmethod
    def print_lateFusionObject_performanceMetrics(kumal_base_folder_location, datasetName, truncatedDuration, logcatRuntimeThreshold, malwarePercent, trainedModelDetails, args):
        """
        Prints all the performance metric stored in the late fusion object
        params:
            - kumal_base_folder_location: Used for accessing the corresponding late stage fusion object
            - datasetName: Used for accessing the corresponding HPC_classifier object
            - truncatedDuration, logcatRuntimeThreshold, malwarePercent: Used for accessing the HPC_classifier object for the cd-dataset
            - trainedModelDetails: Used for accessing the HPC_classifier object
        """
        # Load the late stage fusion object
        datasetType, _ = dataloader_generator.get_dataset_type_and_partition_dist(dataset_name = datasetName)
        if datasetType == 'std-dataset':
            savePath = os.path.join(kumal_base_folder_location, "res", "trainedModels", "std-dataset")
            savePath = os.path.join(savePath,f"HPC_Object_logRuntime{trainedModelDetails['logcatRuntimeThreshold']}_truncDuration{trainedModelDetails['truncatedDuration']}_malwarePercent{trainedModelDetails['malwarePercent']}.pkl")
        elif datasetType == 'cd-dataset':
            savePath = os.path.join(kumal_base_folder_location, "res", "trainedModels", datasetName)
            savePath = os.path.join(savePath,f"HPC_Object_logRuntime{logcatRuntimeThreshold}_truncDuration{truncatedDuration}_malwarePercent{malwarePercent}_trainedModel{trainedModelDetails['logcatRuntimeThreshold']}_{trainedModelDetails['truncatedDuration']}_{trainedModelDetails['malwarePercent']}.pkl")
        else:
            raise ValueError(f"Incorrect dataset name passed: {datasetName}")

        HPC_classifier_inst = HPC_classifier(args=args)
        HPC_classifier_inst.load_HPC_clf_object(fpath=savePath)

        # Pretty print the performance metrics
        HPC_classifier_inst.pretty_print_performance_metric()
        
    @staticmethod
    def perf_vs_lrt(args, kumal_base_folder_location):
        """
        Used for generating the performance vs. lRT plot for the cd-dataset
        """
        basePath_featureEngineeredDataset="/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset"
        datasetName = "cdyear3-dataset"  
        
        # Trained model to be used for testing
        orchInst = orchestrator(args=args, 
                    basePath_featureEngineeredDataset=basePath_featureEngineeredDataset, 
                    datasetName=datasetName, 
                    malwarePercent=0.5,
                    kumal_base_folder_location=kumal_base_folder_location)
    
        # # Grid search all parameter using the trained model.
        # for logcatRuntimeThreshold in orchInst.candidateLogcatRuntimeThresholds:
        #     for truncatedDuration in orchInst.candidateTruncatedDurations:
        #         print(f" ---------- Generating late-stage-fusion instance for logcatRuntimeThreshold {logcatRuntimeThreshold} and truncatedDuration {truncatedDuration} ----------")
                
        #         savedTrainedModelDetails = {"logcatRuntimeThreshold": 0, "truncatedDuration" : truncatedDuration, "malwarePercent":0.5}
        
        #         orchInst.cd_dataset_tasks(trainedModelDetails=savedTrainedModelDetails,
        #                                 logcatRuntimeThreshold=logcatRuntimeThreshold, 
        #                                 truncatedDuration=truncatedDuration, 
        #                                 print_performance_metric=True, 
        #                                 save_HPC_Clf_Object=True)  
                
        
        # Generate the performance vs. lRT plot
        performanceMetricType = {"class_of_interest": "class-1", 
                                "splitType": "testing", 
                                "hpc-group-name": "all", 
                                "performanceMetricName": 'recall'}
        
        trainedModelDetails = None
        orchInst.prettyPrintGridPerformanceMetrics(datasetName = "cdyear1-dataset", 
                                                    performanceMetricType = performanceMetricType,
                                                    malwarePercent=0.5,
                                                    trainedModelDetails=trainedModelDetails)
        
    
    @staticmethod
    def unit_test_orchestrator(args, kumal_base_folder_location):
        basePath_featureEngineeredDataset="/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset"
        orchestrator.perf_vs_lrt(args, kumal_base_folder_location)
        exit()
        # ########################## Testing std-dataset tasks ##########################
        # orchInst = orchestrator(args=args, 
        #                         basePath_featureEngineeredDataset=basePath_featureEngineeredDataset, 
        #                         datasetName="std-dataset", 
        #                         malwarePercent=0.5,
        #                         kumal_base_folder_location=kumal_base_folder_location)
        # orchInst.std_dataset_tasks(logcatRuntimeThreshold=0, truncatedDuration=10, print_performance_metric=True, saveTrainedModels=True, trainHPCClassifiers=True)                
        # exit()
        # ##############################################################################
        
        # ########################## Testing cd-dataset tasks ###########################
        # orchInst = orchestrator(args=args, 
        #                         basePath_featureEngineeredDataset=basePath_featureEngineeredDataset, 
        #                         datasetName="cdyear1-dataset", 
        #                         malwarePercent=0.5,
        #                         kumal_base_folder_location=kumal_base_folder_location)
        # # Trained model to be used for testing
        # trainedModelDetails = {"logcatRuntimeThreshold":0, "truncatedDuration":10, "malwarePercent":0.5}
        # orchInst.cd_dataset_tasks(trainedModelDetails=trainedModelDetails, logcatRuntimeThreshold=0, truncatedDuration=10, print_performance_metric=True, save_HPC_Clf_Object=True)                
        # exit()
        # ################################################################################
        
        
        ######################## Testing grid search task ##########################
        orchInst = orchestrator(args=args, 
                                basePath_featureEngineeredDataset=basePath_featureEngineeredDataset, 
                                datasetName="std-dataset", 
                                malwarePercent=0.5,
                                kumal_base_folder_location=kumal_base_folder_location)
        # trainedModelDetails = {"logcatRuntimeThreshold":75, "truncatedDuration":90, "malwarePercent":0.5}
        trainedModelDetails=None
        orchInst.logcat_runtime_vs_truncated_duration_grid(trainHPCClassifiers=True,
                                                            trainedModelDetails=trainedModelDetails) 
        exit()
        #############################################################################

        # ######################## Print the performance metrics of a late stage fusion object ##########################
        # datasetName = "std-dataset"
        # logcatRuntimeThreshold=0
        # truncatedDuration=10
        # malwarePercent=0.5
        # trainedModelDetails = {"logcatRuntimeThreshold":0, "truncatedDuration":10, "malwarePercent":0.5}

        # orchestrator.print_lateFusionObject_performanceMetrics(kumal_base_folder_location=kumal_base_folder_location, 
        #                                                         datasetName=datasetName, 
        #                                                         truncatedDuration=truncatedDuration, 
        #                                                         logcatRuntimeThreshold=logcatRuntimeThreshold, 
        #                                                         malwarePercent=malwarePercent, 
        #                                                         trainedModelDetails=trainedModelDetails, 
        #                                                         args=args)
        # exit()
        # ###############################################################################################################
        
        ########################## Testing the performance grid generator ##########################
        orchInst = orchestrator(args=args, 
                                basePath_featureEngineeredDataset=basePath_featureEngineeredDataset, 
                                datasetName="std-dataset", 
                                malwarePercent=0.5,
                                kumal_base_folder_location=kumal_base_folder_location)

        ############################################################## To be used as reference ###################################################################################################
        # performanceMetricType (dict): {
        #                                   "class_of_interest": "benign", "malware", or "summary",
        #                                   "splitType": "training" or "testing",
        #                                   "hpc-group-name": "hpc-group-1", "hpc-group-2", "hpc-group-3", "hpc-group-4", or "all"
        #                                   "performanceMetricName": 'f1', 'precision', or 'recall'
        #                               }
        ##########################################################################################################################################################################################

        performanceMetricType = {"class_of_interest": "summary", 
                                "splitType": "testing", 
                                "hpc-group-name": "hpc-group-4", 
                                "performanceMetricName": 'f1'}
        
        # trainedModelDetails = {"logcatRuntimeThreshold":0, "truncatedDuration":90, "malwarePercent":0.5}
        trainedModelDetails = None
                
        orchInst.prettyPrintGridPerformanceMetrics(datasetName = "std-dataset", 
                                                    performanceMetricType = performanceMetricType,
                                                    malwarePercent=0.5,
                                                    trainedModelDetails=trainedModelDetails)
        ############################################################################################





def main_worker(args, kumal_base_folder_location):
    """
    Worker node that performs the complete analysis.
    params:
        - args: easydict storing the experimental parameters
        - kumal_base_folder_location: Location of base folder of xmd
    """
    # resample_dataset.unitTestResampler()
    # baseRFmodel.unit_test_baseRFmodel(args=args)
    # HPC_classifier.unit_test_HPC_classifier(args=args)
    # featureEngineeredDatasetLoader.unit_test_featureEngineeredDatasetLoader()
    orchestrator.unit_test_orchestrator(args=args, kumal_base_folder_location= kumal_base_folder_location)


def main():
    ############################################## Setting up the experimental parameters ##############################################
    # Location of the base folder of xmd 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    base_folder_location = os.path.join(dir_path.replace("/src",""),"")

    # Get the arguments and the config file object
    cfg, args = get_args(kumal_base_folder_location=base_folder_location)

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
    main_worker(args=args, kumal_base_folder_location= base_folder_location)

if __name__ == '__main__':
    main()