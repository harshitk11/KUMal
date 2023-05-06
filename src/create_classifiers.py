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


BENIGN_LABEL = 0
MALWARE_LABEL = 1

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
        df_wide = df.pivot(index='truncatedDuration', columns='logcatRuntimeThreshold', values='performanceMetric')
        sns.set_style("ticks")
        sns.set(font_scale=1.2)
        sns.set(rc={'figure.figsize':(10,7)})
        sns.lineplot(data=df_wide, markers=True, markersize=15)

        plt.rcParams["font.weight"] = "bold"
        plt.xticks(weight="bold")
        plt.yticks(weight="bold")
        plt.title(plotTitle, weight="bold")
        plt.xlabel("Truncated Duration", weight="bold")
        plt.ylabel(performanceMetric, weight="bold")
        plt.legend(title="logcatRuntimeThreshold", prop={'weight': 'bold'})
        plt.ylim(0.3,0.7)
        sns.despine(top=True, right=True)
        plt.tight_layout()
        if saveLocation:
            plt.savefig(saveLocation, dpi=300)   


class performanceMetricAggregator:
    @staticmethod
    def generatePerformanceMetricTypeDict(datasetName, performanceMetricDict, performanceMetricName, selectedBaseClassifier=None, globlChannelType=None, hpc_group_name=None, clfToi=None):
        """
        Generates the dict that is used by getAggregatePerformanceMetric()
        """
        datasetType, _ = dataloader_generator.get_dataset_type_and_partition_dist(dataset_name = datasetName)
        performanceMetricType = {}

        performanceMetricType["performanceMetricDict"] = performanceMetricDict
        if datasetType == "std-dataset":
            performanceMetricType["splitType"] = "training"
        elif datasetType == "cd-dataset":
            performanceMetricType["splitType"] = "testing"
        else:
            raise ValueError(f"DatasetType not supported: {datasetType}")

        performanceMetricType["selectedBaseClassifier"] = selectedBaseClassifier
        performanceMetricType["globlChannelType"] = globlChannelType
        performanceMetricType["hpc-group-name"] = hpc_group_name
        performanceMetricType["clfToi"] = clfToi
        performanceMetricType["performanceMetricName"] = performanceMetricName

        return performanceMetricType

    @staticmethod
    def getAggregatePerformanceMetric(lateFusionInstance, performanceMetricType):
        """
        Returns a single performance metric from the lateFusionInstance based on the performanceMetricType.

        params:
            - lateFusionInstance (late_stage_fusion): Instance of the late_stage_fusion object with all the performance metrics updated.
            - performanceMetricType (dict): {
                                        "performanceMetricDict": "stage1ClassifierPerformanceMetrics", "globlFusionPerformanceMetric", or "hpcGloblFusionPerformanceMetricAllGroup",
                                        "splitType": "training" or "testing",
                                        "selectedBaseClassifier": "globl", "hpc", or "all"
                                        "globlChannelType": "globl", or "dvfs",
                                        "hpc-group-name": "hpc-group-1", "hpc-group-2", "hpc-group-3", or "hpc-group-4",
                                        "clfToi": "hpc", "hpc-dvfs-ensemble", "hpc-dvfs-sg-rf", "hpc-dvfs-sg-lr", "hpc-globl-ensemble", "hpc-globl-sg-rf", or "hpc-globl-sg-lr"
                                        "performanceMetricName": 'f1', 'precision', or 'recall'
                                    }
                            NOTE: depending on the performanceMetricDict, only some of the fields in the dict are required:
                                (1) if "performanceMetricDict": "stage1ClassifierPerformanceMetrics"
                                        Need: splitType, selectedBaseClassifier, performanceMetricName
                                (2) if "performanceMetricDict": "globlFusionPerformanceMetric"
                                        Need: splitType, globlChannelType, performanceMetricName
                                (3) if "performanceMetricDict": "hpcGloblFusionPerformanceMetricAllGroup"
                                        Need: splitType, hpc-group-name, clfToi, performanceMetricName
        Output:
            - performanceMetric (int)
        """
        performanceMetricType_ = performanceMetricType["performanceMetricDict"]
        
        if performanceMetricType_ == "stage1ClassifierPerformanceMetrics":
            splitType = performanceMetricType["splitType"]
            selectedBaseClassifier = performanceMetricType["selectedBaseClassifier"]
            performanceMetricName = performanceMetricType["performanceMetricName"]
            performancMetricScoreList = []
            
            if selectedBaseClassifier == "globl":
                for chnName in late_stage_fusion.globlChannelNameList:
                    pScore = lateFusionInstance.stage1ClassifierPerformanceMetrics[splitType][chnName][performanceMetricName]
                    performancMetricScoreList.append(pScore)

            elif selectedBaseClassifier == "hpc":
                for grpName in late_stage_fusion.hpcGroupNameList:
                    pScore = lateFusionInstance.stage1ClassifierPerformanceMetrics[splitType][grpName][performanceMetricName]
                    performancMetricScoreList.append(pScore)
                
            elif selectedBaseClassifier == "all":
                for chnGrpName in (late_stage_fusion.globlChannelNameList+late_stage_fusion.hpcGroupNameList):
                    pScore = lateFusionInstance.stage1ClassifierPerformanceMetrics[splitType][chnGrpName][performanceMetricName]
                    performancMetricScoreList.append(pScore)

            else:
                raise ValueError(f"Incorrect selectedBaseClassifier : {performanceMetricType['selectedBaseClassifier']}")

            # Get the mean of the performanceMetric
            aggregatePerformanceMetric = sum(performancMetricScoreList)/len(performancMetricScoreList)
            return aggregatePerformanceMetric

        elif performanceMetricType_ == "globlFusionPerformanceMetric":
            splitType = performanceMetricType["splitType"]
            globlChannelType = performanceMetricType["globlChannelType"]
            performanceMetricName = performanceMetricType["performanceMetricName"]
            pScore = lateFusionInstance.globlFusionPerformanceMetric[splitType][globlChannelType][performanceMetricName]
            return pScore

        elif performanceMetricType_ == "hpcGloblFusionPerformanceMetricAllGroup":
            splitType = performanceMetricType["splitType"]
            hpc_group_name = performanceMetricType["hpc-group-name"]
            clfToi = performanceMetricType["clfToi"]
            performanceMetricName = performanceMetricType["performanceMetricName"]
            pScore = lateFusionInstance.hpcGloblFusionPerformanceMetricAllGroup[splitType][hpc_group_name][clfToi][performanceMetricName]
            return pScore
            
        else:
            raise ValueError(f"Incorrect performanceMetricDict : {performanceMetricType_}")

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

    def resampleBaseTensor(self, X, y):
        """
        Resamples the Hpc Tensor.
        params:
            - X: dataset (Nchannels, Nsamples, feature_size)
            - y: labels (Nsamples,) 
        Output:
            - X_res, y_res : Resampled dataset
        """
        # Get the sampling indices
        sampIndx = self.generate_sampling_indices(X[0], y)

        # Resample the dataset
        X_res = X[:,sampIndx,:]
        y_res = y[sampIndx]

        return X_res, y_res

    def resampleHpcTensor(self, Xlist, yList):
        """
        Resamples the dataset for all the HPC groups.
        
        params:
            - Xlist: [dataset_group1, dataset_group2, dataset_group3, dataset_group4] | dataset shape: (1, Nsamples, feature_size)
            - yList: [labels_group1, labels_group2, labels_group3, labels_group4] | labels_shape: (Nsamples,)

        Output:
            - Xlist_res, yList_res : Resampled dataset
        """
        Xlist_res = []
        yList_res = []

        for grpIndx, _ in enumerate(Xlist):
            X_res, y_res = self.resampleBaseTensor(X=Xlist[grpIndx], y=yList[grpIndx])
            Xlist_res.append(X_res)
            yList_res.append(y_res)

        return Xlist_res, yList_res

    @staticmethod
    def unitTestResampler():
        # Loading the datasets for testing the fusion modules
        hpc_x_test = []
        hpc_y_test = []
        for group in ["rn1","rn2","rn3","rn4"]:
            hpc_x_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset/std-dataset/0/10/{group}/channel_bins_train.npy"))
            hpc_y_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/KUMal/data/featureEngineeredDataset/std-dataset/0/10/{group}/labels_train.npy"))
      
        print(" [Pre] Details of HPC data")
        print(f" - Shape of the hpc data and label : {[(dataset.shape,labels.shape) for dataset,labels in zip(hpc_x_test,hpc_y_test)]}")
        print(" - [Pre] Number of malware and benign samples")
        print([Counter(Y) for Y in hpc_y_test])
        
        rmInst = resample_dataset(malwarePercent=0.1)
        hpc_x_test, hpc_y_test = rmInst.resampleHpcTensor(Xlist=hpc_x_test, yList=hpc_y_test)
        
        print(" -------------------------------------------------------------------------------------------------")
        print(" [Post] Details of resampled HPC data")
        print(f" - Shape of the hpc data and label : {[(dataset.shape,labels.shape) for dataset,labels in zip(hpc_x_test,hpc_y_test)]}")
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

            test_performance_metric = {"summary": {'f1': class_results['macro avg']['f1-score'],
                                        'precision': class_results['macro avg']['precision'],
                                        'recall': class_results['macro avg']['recall']},
                                       
                                       "class-1": {'f1': class_results['1']['f1-score'],
                                        'precision': class_results['1']['precision'],
                                        'recall': class_results['1']['recall']},
                                       
                                       "class-0": {'f1': class_results['0']['f1-score'],
                                        'precision': class_results['0']['precision'],
                                        'recall': class_results['0']['recall']}}

            if print_performance_metric:
                # Print the classification report and the confusion matrix
                print(f" ----- Evaluation performation metrics -----")
                print(classification_report(Ytest,predict_labels))
                print(confusion_matrix(Ytest, predict_labels))

        return predict_labels, test_performance_metric 

    def eval_uncertainty(self, Xtest, savePath_boxPlot=None):
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
            print(f"  - Evaluating classifier {i+1}/{num_classifiers}")
            predictions = classifier.predict(Xtest)
            prediction_array[:, i] = predictions

        # Get the probability of each class for each sample
        histogram_array = np.column_stack((np.sum(prediction_array == 0, axis=1), np.sum(prediction_array == 1, axis=1))) 
        uncertainty_score_array = entropy(histogram_array, axis=1, base=2)
        predict_proba = np.mean(prediction_array, axis=1)
    
        if savePath_boxPlot:
            plt.boxplot(predict_proba)
            plt.xlabel('Samples')
            plt.ylabel('Uncertainty Score')
            plt.title('Box Plot of Uncertainty Scores')
            plt.savefig(savePath_boxPlot, dpi=300)
            plt.close()
            
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
    def generate_roc_curve(Ytest, Ypred_proba, fpr_threshold, savePath=None):
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(Ytest, Ypred_proba)
        roc_auc = auc(fpr, tpr)

        # Find the threshold closest to the fpr_threshold
        decision_threshold = thresholds[np.argmin(np.abs(fpr - fpr_threshold))]
        print([(i,j,k) for i,j,k in zip(fpr, tpr, thresholds)])
        print(decision_threshold)
        
        plt.plot(fpr, tpr, color='teal', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
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
        
        return decision_threshold
        
    @staticmethod
    def unit_test_baseRFmodel(args):
        """
        unit test for the base RF model object
        """
        # Loading the datasets for testing the fusion modules
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
        test_uncertainty, predict_proba = newBaseModelInst.eval_uncertainty(Xtest=hpc_x_test[0], savePath_boxPlot="test.png")
        print(np.mean(test_uncertainty))
        
        baseRFmodel.generate_roc_curve(Ytest=hpc_y_test, Ypred_proba=predict_proba, savePath="test2.png", fpr_threshold=0.2)
        # newBaseModelInst.eval(Xtest=X_test[0],Ytest=Y_test,print_performance_metric=True)

class late_stage_fusion:
    """
    Object for tracking the stage-1 and stage-2 predictive performance.
    """        
    hpcGroupNameList = ["hpc-group-1", "hpc-group-2", "hpc-group-3", "hpc-group-4"]

    def __init__(self, args) -> None:
        self.args = args
        # Stores a dict of baseRFmodel objects. One for every hpc group. {grpName: baseRFModel object, ...}
        self.hpcGroupBaseClf = None
        
        ########################################## Performance metricts  ##########################################
        # List of file hashes used for training the base classifiers
        self.baseClassifierTrainFileHashList = None 
         
        # List of performance metrics for all the base classifiers of GLOBL channels and HPC groups. 
        stage1ClfPerformanceMetricTemplate = {chnGrpName: None for chnGrpName in late_stage_fusion.hpcGroupNameList}
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
                
                baseModelInst = baseRFmodel(args=self.args, channelName=groupName, channelType="hpc")
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
        
    
    def pretty_print_performance_metric(self, baseClfPerfFlag=False):
        """
        Prints the performance metric to stdout.
        """
        if baseClfPerfFlag:
            print("\n----------- Base classifiers performance metric -----------")
            for splitType, stage1ClfPerformanceMetricTemplate in self.stage1ClassifierPerformanceMetrics.items():
                print(f"----------- Split Type : {splitType} -----------")
                for chnGrpName, perfMetric in stage1ClfPerformanceMetricTemplate.items():
                    try:
                        print(f" Channel name: {chnGrpName} | F1 : {perfMetric['f1']} | precision : {perfMetric['precision']} | recall : {perfMetric['recall']}")
                    except:
                        print(f" Channel name: {chnGrpName} | *********** ")

       
    @staticmethod
    def get_hashList_from_fileList(file_paths):
        """
        Returns a list of hashes from the file list.
        params:
            - file_paths (list): List of file paths

        Output:
            - hashList (list): List of hashes extracted from the file paths
        """
        hashList = []
        regex_pattern = r'.*\/(.*)__.*'

        # Parse this file list to extract the hashes
        for filename in file_paths:
            file_hash_obj = re.search(regex_pattern, filename, re.M|re.I)
            file_hash_string = file_hash_obj.group(1).strip()
            hashList.append(file_hash_string)

        return hashList

    def save_fusion_object(self, fpath):
        """
        Saves the model and the model details to the specified path.

        params:
            - fpath: full path where the model should be saved
        """
        # Used for saving and loading the object
        model_pickle_file = {   "hpcGroupBaseClf":self.hpcGroupBaseClf,
                                "baseClassifierTrainFileHashList":self.baseClassifierTrainFileHashList,
                                "stage1ClassifierPerformanceMetrics":self.stage1ClassifierPerformanceMetrics,  
                            }

        # Write the pickle file
        with open(fpath, 'wb') as handle:
            pickle.dump(model_pickle_file, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        

    def load_fusion_object(self, fpath):
        """
        Loads the model from the specified path. And populates all the corresponding model details of this object.
        """
        # Load the dict from fpath and update the instance attributes
        with open(fpath, 'rb') as handle:
            model_pickle_file = pickle.load(handle)

        self.hpcGroupBaseClf = model_pickle_file["hpcGroupBaseClf"]
        self.baseClassifierTrainFileHashList = model_pickle_file["baseClassifierTrainFileHashList"]
        self.stage1ClassifierPerformanceMetrics = model_pickle_file["stage1ClassifierPerformanceMetrics"]
        

    @staticmethod
    def unit_test_lateStageFusion(args):
        """
        unit test for the late_stage_fusion object
        """
        ############################### Testing the base classifier modules ###############################
        # Loading the unmatched datasets for the unit tests
        hpc_x_train = []
        hpc_y_train = []
        hpc_x_test = []
        hpc_y_test = []
        for group in ["rn1","rn2","rn3","rn4"]:
            hpc_x_train.append(np.load(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/std-dataset/15/40/HPC_individual/{group}/channel_bins_train.npy"))
            hpc_y_train.append(np.load(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/std-dataset/15/40/HPC_individual/{group}/labels_train.npy"))
            hpc_x_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/15/40/HPC_individual/{group}/channel_bins_test.npy"))
            hpc_y_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/15/40/HPC_individual/{group}/labels_test.npy"))
        
        print("Details of HPC training and test data")
        print(f" - Shape of the training data and label : {[(dataset.shape,labels.shape) for dataset,labels in zip(hpc_x_train,hpc_y_train)]}")
        print(f" - Shape of the test data and label : {[(dataset.shape,labels.shape) for dataset,labels in zip(hpc_x_test,hpc_y_test)]}")
        
        #################### Resampler ####################
        rmInst = resample_dataset(malwarePercent=0.5)
        hpc_x_train, hpc_y_train = rmInst.resampleHpcTensor(Xlist=hpc_x_train, yList=hpc_y_train)
        hpc_x_test, hpc_y_test = rmInst.resampleHpcTensor(Xlist=hpc_x_test, yList=hpc_y_test)
        ###################################################
        
        # # Testing the training module
        # print(f" - Training the hpc and globl base classifiers -")
        # lateFusionInstance = late_stage_fusion(args=args)
        # lateFusionInstance.stage1trainHPC(XtrainHPC=hpc_x_train, YtrainHPC=hpc_y_train, updateObjectPerformanceMetrics=True)
        
        # print(" - Summary performance metric of all the models [post training] -")
        # lateFusionInstance.pretty_print_performance_metric(baseClfPerfFlag=True)

        # # Testing the evaluation module
        # print(f" - Evaluating the hpc and globl base classifiers -")
        # allGroupPredictions = lateFusionInstance.stage1evalHPC(XtestHPC=hpc_x_test, YtestHPC=hpc_y_test, updateObjectPerformanceMetrics=True, print_performance_metric=True)

        # print(" - Shape of the predicted labels post evaluation -")
        # print(f" - HPC predicted labels for all groups (Nsamples, ): {[grp.shape for grp in allGroupPredictions]}")

        # print(" - Summary performance metric of all the models [post evaluation] -")
        # lateFusionInstance.pretty_print_performance_metric(baseClfPerfFlag=True)
        
        # # Testing the loading and saving module
        # print(f" - Saving the object -")
        # lateFusionInstance.save_fusion_object(fpath="testmodel.pkl")
        
        # print(f" - Loading the object and testing the models -")
        # lateFusionInstance = late_stage_fusion(args=args)
        # lateFusionInstance.load_fusion_object(fpath="testmodel.pkl")
        # lateFusionInstance.stage1evalHPC(XtestHPC=hpc_x_test, YtestHPC=hpc_y_test, updateObjectPerformanceMetrics=False, print_performance_metric=True)

        # print(" - Summary performance metric of all the models [post evaluation post loading] -")
        # lateFusionInstance.pretty_print_performance_metric(baseClfPerfFlag=True)
        # exit()
        


class featureEngineeredDatasetLoader:
    """
    Contains all the helper methods used for loading the feature engineered dataset.
    """
    def __init__(self, basePath_featureEngineeredDataset, datasetName, logcatRuntimeThreshold, truncatedDuration):
        """
        Loads the dataset for all classification tasks of interest.

        params:
            - basePath_featureEngineeredDataset (str): Location of the base folder where all the feature engineered dataset is stored
            - datasetName (str): Can be one of the following {'std-dataset', 'bench-dataset', 'cdyear1-dataset', 'cdyear2-dataset', 'cdyear3-dataset'}
            - logcatRuntimeThreshold (int), truncatedDuration (int) : Used for accessing the corresponding dataset
        
        Output:
            - featEngineeredData (dict): {'DVFS_partition_for_HPC_DVFS_fusion':{"train": ... , "trainSG": ... , "test": ...},
                                                'HPC_partition_for_HPC_DVFS_fusion':{"train": ... , "trainSG": ... , "test": ...},
                                                'HPC_individual':{"train": ... , "trainSG": ... , "test": ... },
                                                'DVFS_individual':{"train": ... , "trainSG": ... , "test": ...},
                                                'DVFS_fusion':{"train": ... , "trainSG": ... , "test": ...} 
                                                
                                                ... -> [X_list, y_list, file_list] or [X_dat, y_dat, file_dat] 
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
            - featEngineeredData (dict): {'DVFS_partition_for_HPC_DVFS_fusion':{"train": ... , "trainSG": ... , "test": ...},
                                                'HPC_partition_for_HPC_DVFS_fusion':{"train": ... , "trainSG": ... , "test": ...},
                                                'HPC_individual':{"train": ... , "trainSG": ... , "test": ... },
                                                'DVFS_individual':{"train": ... , "trainSG": ... , "test": ...},
                                                'DVFS_fusion':{"train": ... , "trainSG": ... , "test": ...} 
                                                
                                                ... -> [X_list, y_list, file_list] or [X_dat, y_dat, file_dat]
        """
        featureEngineeredData = {}

        for clfToi, partitionActivationFlag in self.requiredPartitions.items():
            featureEngineeredData[clfToi] = {}
            for partition, activationFlag in partitionActivationFlag.items():
                featureEngineeredData[clfToi][partition] = None
                if not activationFlag:
                    continue
    
                # Else load the dataset for this partition
                if clfToi in ["DVFS_partition_for_HPC_DVFS_fusion", "HPC_partition_for_HPC_DVFS_fusion", "HPC_individual"]:
                    featureEngineeredData[clfToi][partition] = self.load_rnBucket_dataset(clf_toi=clfToi, partition_type=partition)
                elif clfToi in ["DVFS_individual", "DVFS_fusion"]:
                    featureEngineeredData[clfToi][partition] = self.load_globlOnlyType_dataset(clf_toi=clfToi, partition_type=partition)
                else:
                    raise ValueError(f"Incorrect classification task {clfToi}")
        
        return featureEngineeredData
                    
    def load_rnBucket_dataset(self, clf_toi, partition_type):
        """
        Used for loading partitions that are divided into rn buckets.

        params:
            - clf_toi: DVFS_partition_for_HPC_DVFS_fusion, HPC_partition_for_HPC_DVFS_fusion, or HPC_individual
            - partition_type: "train", "trainSG", or "test"

        Output: X_list, y_list, file_list
            - X_list (ndarray): [dataset_group1, dataset_group2, dataset_group3, dataset_group4] | dataset shape: (1, Nsamples, feature_size)
            - y_list (ndarray): [labels_group1, labels_group2, labels_group3, labels_group4] | labels_shape: (Nsamples,)
            - file_list (ndarray): [files_group1, files_group2, files_group3, files_group4] | files_shape: (Nsamples,)
        """
        X_list, y_list, file_list = [], [], []
        for group in ["rn1","rn2","rn3","rn4"]:
            X_list.append(np.load(os.path.join(self.basePath_featureEngineeredDataset, self.datasetName, str(self.logcatRuntimeThreshold), str(self.truncatedDuration), clf_toi, group, f"channel_bins_{partition_type}.npy")))
            y_list.append(np.load(os.path.join(self.basePath_featureEngineeredDataset, self.datasetName, str(self.logcatRuntimeThreshold), str(self.truncatedDuration), clf_toi, group, f"labels_{partition_type}.npy")))
            with open(os.path.join(self.basePath_featureEngineeredDataset, self.datasetName, str(self.logcatRuntimeThreshold), str(self.truncatedDuration), clf_toi, group, f"file_paths_{partition_type}.npy"), 'rb') as fp:
                file_list.append(np.array(pickle.load(fp), dtype="object"))
            
        return X_list, y_list, file_list

    def load_globlOnlyType_dataset(self, clf_toi, partition_type):
        """
        Used for loading partitions that have not been divided into rn buckets.

        params:
            - clf_toi: DVFS_individual, or DVFS_fusion
            - partition_type: "train", "trainSG", or "test"

        Output: X_dat, y_dat, file_dat
            - X_dat (ndarray): dataset (Nchannels, Nsamples, feature_size)
            - y_dat (ndarray): labels (Nsamples,)
            - file_dat (ndarray): list of corresponding file paths (Nsamples,)
        """
        X_dat = np.load(os.path.join(self.basePath_featureEngineeredDataset, self.datasetName, str(self.logcatRuntimeThreshold), str(self.truncatedDuration), clf_toi, "all", f"channel_bins_{partition_type}.npy"))
        y_dat = np.load(os.path.join(self.basePath_featureEngineeredDataset, self.datasetName, str(self.logcatRuntimeThreshold), str(self.truncatedDuration), clf_toi, "all", f"labels_{partition_type}.npy"))
        with open(os.path.join(self.basePath_featureEngineeredDataset, self.datasetName, str(self.logcatRuntimeThreshold), str(self.truncatedDuration), clf_toi, "all", f"file_paths_{partition_type}.npy"), 'rb') as fp:
            file_dat = np.array(pickle.load(fp), dtype="object")
             
        return X_dat, y_dat, file_dat
    
    @staticmethod
    def unit_test_featureEngineeredDatasetLoader():
        loadDatasetInst = featureEngineeredDatasetLoader(basePath_featureEngineeredDataset="/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset",
                                                        datasetName="std-dataset",
                                                        logcatRuntimeThreshold=5,
                                                        truncatedDuration=10)
        featureEngineeredData = loadDatasetInst.load_dataset()
        for clfToi, partitionActivationFlag in featureEngineeredData.items():
            print(f"Classification task of interest: {clfToi}")
            for partitionName, partition in partitionActivationFlag.items():
                if partition is None:
                    print(f"Partition- {partitionName} | *********** ")
                    continue
    
                if clfToi in ["DVFS_partition_for_HPC_DVFS_fusion", "HPC_partition_for_HPC_DVFS_fusion", "HPC_individual"]:
                    print(f"Partition- {partitionName} | Size: {[(x.shape,y.shape,z.shape) for x,y,z in zip(partition[0],partition[1],partition[2])]}")
                elif clfToi in ["DVFS_individual", "DVFS_fusion"]:
                    print(f"Partition- {partitionName} | Size: {partition[0].shape,partition[1].shape,partition[2].shape}")
                    

class orchestrator:
    """
    Orchestrates the training and evaluation tasks for all the datasets and classification tasks of interest.
    """
    
    def __init__(self, args, basePath_featureEngineeredDataset, datasetName, malwarePercent, xmd_base_folder_location) -> None:
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
        self.xmd_base_folder_location = xmd_base_folder_location

        # 2-D array storing the grid containing paths of saved late-stage-fusion objects
        ## self.lateStageFusionObject : {<logcatRuntimeThreshold>:{<truncatedDuration>: ... }, ... }
        self.lateStageFusionObject = {}
        for logcatRuntimeThreshold in self.candidateLogcatRuntimeThresholds:
            self.lateStageFusionObject[logcatRuntimeThreshold] = {}
            for truncatedDuration in self.candidateTruncatedDurations:
                self.lateStageFusionObject[logcatRuntimeThreshold][truncatedDuration] = None


    def std_dataset_tasks(self, logcatRuntimeThreshold, truncatedDuration, print_performance_metric, saveTrainedModels, trainStage1, trainStage2):
        """
        Tasks:
            -Load the dataset for all the classification tasks.
            -Resample the dataset.
            -Train the HPC base classifiers.
            -Train the DVFS base classifiers.
            -Save the hash list used for training the base classifiers
            -Save the trained base classifiers.
        params:
            - logcatRuntimeThreshold (int): Filter runtime threshold used for filtering the dataset.
            - truncatedDuration (int): To truncate the time series, i.e., take the first truncatedDuration seconds.
            - print_performance_metric (bool): If True, then print the performance metrics for all evaluations.
            - saveTrainedModels (bool): If True, then save the lateStageFusion object in the trainedModels folder.
            - trainStage1 (bool): Train stage 1 models if True.
            - trainStage2 (bool): Train stage 2 models if True.
        Output:
            - lateFusionInstance : Instance of the object storing all the trained models
            - self.lateStageFusionObject : Stores the path of late-stage-fusion object for the corresponding logcatRuntimeThreshold and truncatedDuration.
        """
        ######################################### Load the dataset for all the tasks ########################################
        assert self.datasetName == "std-dataset", "Incorrect dataset name"
        loadDatasetInst = featureEngineeredDatasetLoader(basePath_featureEngineeredDataset=self.basePath_featureEngineeredDataset,
                                                        datasetName=self.datasetName,
                                                        logcatRuntimeThreshold=logcatRuntimeThreshold,
                                                        truncatedDuration=truncatedDuration)
        featureEngineeredData = loadDatasetInst.load_dataset()

        # For training the globl and hpc base classifiers
        hpc_X_train, hpc_Y_train, hpc_fileList = featureEngineeredData['HPC_individual']['train']
        ######################################################################################################################
        
        ######################################## Resampler ########################################
        rmInst = resample_dataset(malwarePercent=self.malwarePercent)
        hpc_X_train, hpc_Y_train = rmInst.resampleHpcTensor(Xlist=hpc_X_train, yList=hpc_Y_train)
        ###########################################################################################
        
        lateFusionInstance = None
        if trainStage1:
            ######################################### Train the HPC and the DVFS base classifiers ########################################
            print(f" - Training the hpc and globl base classifiers -")
            lateFusionInstance = late_stage_fusion(args=self.args)
            lateFusionInstance.stage1trainHPC(XtrainHPC=hpc_X_train, YtrainHPC=hpc_Y_train, updateObjectPerformanceMetrics=False)
            ##############################################################################################################################
            
            ################################### Update the performance metrics using the train-SG dataset ################################
            XtrainSG = lateFusionInstance.generateTrainSGdataset(XtestGLOBL_HPC=XtestGLOBL_HPC_res, YtestGLOBL_HPC = YtestGLOBL_HPC_res)
        
            # Update the base classifier scores
            lateFusionInstance.update_stage1_performance_metrics_USING_TRAINSG(XtrainSG=XtrainSG, 
                                                                                updateObjectPerformanceMetrics=True, 
                                                                                print_performance_metric=print_performance_metric)
            # Generate the globl and dvfs fusion scores
            lateFusionInstance.stage2_globlFusion_ensemble_eval_USING_TRAINSG(XtrainSG = XtrainSG, 
                                                                            updateObjectPerformanceMetrics=True, 
                                                                            print_performance_metric=print_performance_metric)
            
            # Generate the hpc-globl fusion scores
            lateFusionInstance.stage2_hpcGlobl_ensemble_eval(XtrainSG = XtrainSG, 
                                                            updateObjectPerformanceMetrics=True, 
                                                            print_performance_metric=print_performance_metric,
                                                            splitType="training")
            #####################################################################################################################################
            # Save the hash list used for training the base classifiers
            lateFusionInstance.baseClassifierTrainFileHashList = late_stage_fusion.get_hashList_from_fileList(file_paths=dvfs_fileList)
            
            if print_performance_metric:
                # Pretty print performance metric
                print(f"Summary of performance metrics for dataset : {self.datasetName}, logcatRuntimeThreshold : {logcatRuntimeThreshold}, truncatedDuration : {truncatedDuration}")
                lateFusionInstance.pretty_print_performance_metric(baseClfPerfFlag=True)
        
            # Save the trained models
            if saveTrainedModels:
                savePath = os.path.join(self.xmd_base_folder_location, "res", "trainedModels", self.datasetName)
                if not os.path.isdir(savePath):
                    os.makedirs(savePath)
                savePath = os.path.join(savePath,f"lateFusion_logRuntime{logcatRuntimeThreshold}_truncDuration{truncatedDuration}_malwarePercent{self.malwarePercent}.pkl")
                self.lateStageFusionObject[logcatRuntimeThreshold][truncatedDuration] = savePath
                lateFusionInstance.save_fusion_object(fpath=savePath)
        
        else:
            # Load the saved object containing the trained models
            savePath = os.path.join(self.xmd_base_folder_location, "res", "trainedModels", self.datasetName)
            savePath = os.path.join(savePath,f"lateFusion_logRuntime{logcatRuntimeThreshold}_truncDuration{truncatedDuration}_malwarePercent{self.malwarePercent}.pkl")
            lateFusionInstance = late_stage_fusion(args=self.args)
            lateFusionInstance.load_fusion_object(fpath=savePath)

              
        return lateFusionInstance

    def cd_dataset_tasks(self, trainedModelDetails, logcatRuntimeThreshold, truncatedDuration, print_performance_metric, saveLateStageFusionObject):
        """
        Tasks:
            -Load the trained model.
            -Load the dataset for all the classification tasks.
            -Resample the dataset.
            -Test the DVFS base classifier
            -Test HPC base classifier
            -Test Global fusion
            -Test HPC-DVFS fusion
            -Save the hash list used for testing the classifiers

        params:
            - trainedModelDetails (dict) : {"logcatRuntimeThreshold": (int), "truncatedDuration": (int), "malwarePercent": (float)} 
                                        -> Used for loading the corresponding trained model
            - logcatRuntimeThreshold (int), truncatedDuration (int): Used for loading the dataset for the classification task
            - print_performance_metric (bool): If True, then print the performance metrics for all evaluations.
            - saveLateStageFusionObject (bool): If True, then save the updated lateStageFusion object.
        
        Output:
            - lateFusionInstance : Instance of the object storing all the update performance evaluation metrics        
        """
        ################################### Load the trained model ###################################
        savePath = os.path.join(self.xmd_base_folder_location, "res", "trainedModels", "std-dataset")
        savePath = os.path.join(savePath,f"lateFusion_logRuntime{trainedModelDetails['logcatRuntimeThreshold']}_truncDuration{trainedModelDetails['truncatedDuration']}_malwarePercent{trainedModelDetails['malwarePercent']}.pkl")
        
        lateFusionInstance = late_stage_fusion(args=self.args)
        lateFusionInstance.load_fusion_object(fpath=savePath)
        ##############################################################################################

        ################################### Load the dataset for all classification tasks ###################################
        assert (self.datasetName == "cd-dataset") or (self.datasetName == "cdyear1-dataset") or (self.datasetName == "cdyear2-dataset") or (self.datasetName == "cdyear3-dataset"), "Incorrect dataset name"
        loadDatasetInst = featureEngineeredDatasetLoader(basePath_featureEngineeredDataset=self.basePath_featureEngineeredDataset,
                                                        datasetName=self.datasetName,
                                                        logcatRuntimeThreshold=logcatRuntimeThreshold,
                                                        truncatedDuration=truncatedDuration)
        featureEngineeredData = loadDatasetInst.load_dataset()

        # For testing the globl and hpc base classifiers
        dfvs_X_test, dfvs_Y_test, dvfs_fileList = featureEngineeredData['DVFS_individual']['test']
        hpc_X_test, hpc_Y_test, hpc_fileList = featureEngineeredData['HPC_individual']['test']

        # XTrainSG dataset generation
        hpc_X_trainSG, hpc_Y_trainSG, hpc_fileListSG = featureEngineeredData['HPC_partition_for_HPC_DVFS_fusion']['test']
        globl_X_trainSG, globl_Y_trainSG, globl_fileListSG = featureEngineeredData['DVFS_partition_for_HPC_DVFS_fusion']['test']
        XtestGLOBL_HPC= {"globl":globl_X_trainSG, "hpc":hpc_X_trainSG}
        YtestGLOBL_HPC= {"globl":globl_Y_trainSG, "hpc":hpc_Y_trainSG}
        #####################################################################################################################

        ######################################## Resampler ########################################
        rmInst = resample_dataset(malwarePercent=self.malwarePercent)
        dfvs_X_test, dfvs_Y_test = rmInst.resampleBaseTensor(X=dfvs_X_test, y=dfvs_Y_test)
        hpc_X_test, hpc_Y_test = rmInst.resampleHpcTensor(Xlist=hpc_X_test, yList=hpc_Y_test)
        XtestGLOBL_HPC_res, YtestGLOBL_HPC_res = rmInst.resampleFusionTensor(XtestGLOBL_HPC=XtestGLOBL_HPC, YtestGLOBL_HPC=YtestGLOBL_HPC)
        ###########################################################################################

        ################################### Test the HPC and DVFS base classifiers ###################################
        print(f" - Testing the hpc and globl base classifiers -")
        lateFusionInstance.stage1evalGLOBL(XtestGLOBL=dfvs_X_test, YtestGLOBL=dfvs_Y_test, updateObjectPerformanceMetrics=True, print_performance_metric=print_performance_metric)
        lateFusionInstance.stage1evalHPC(XtestHPC=hpc_X_test, YtestHPC=hpc_Y_test, updateObjectPerformanceMetrics=True, print_performance_metric=print_performance_metric)
        ##############################################################################################################

        ################################### Test the GLOBL and HPC-GLOBL fusion ###################################
        print(f" - Testing the globl fusion and hpc-globl fusion -")
        # Testing the globl fusion module
        lateFusionInstance.stage2_globlFusion_ensemble_eval(Xeval=dfvs_X_test, 
                                                            Yeval=dfvs_Y_test, 
                                                            updateObjectPerformanceMetrics=True, 
                                                            print_performance_metric=print_performance_metric,
                                                            splitType="testing")

        XtrainSG = lateFusionInstance.generateTrainSGdataset(XtestGLOBL_HPC=XtestGLOBL_HPC_res, YtestGLOBL_HPC = YtestGLOBL_HPC_res)
        # Generate the hpc-globl fusion scores
        lateFusionInstance.stage2_hpcGlobl_ensemble_eval(XtrainSG = XtrainSG, 
                                                        updateObjectPerformanceMetrics=True, 
                                                        print_performance_metric=print_performance_metric,
                                                        splitType="testing")
        # Generate the hpc-globl-rf scores
        lateFusionInstance.stage2_Eval(XtrainSG, secondStage_modelType='RF', updateObjectPerformanceMetrics=True, print_performance_metric=True)
        ###########################################################################################################
        
        # Save the hash list used for testing the hpc-globl fusion classifiers
        for grpIndx, (groupName, fileHashList) in enumerate(lateFusionInstance.hpcGloblTestFileHashlist.copy().items()):
            lateFusionInstance.hpcGloblTestFileHashlist[groupName] = late_stage_fusion.get_hashList_from_fileList(file_paths=globl_fileListSG[grpIndx])
        
        if print_performance_metric:
            print(f"Summary of performance metrics for dataset : {self.datasetName}, logcatRuntimeThreshold : {logcatRuntimeThreshold}, truncatedDuration : {truncatedDuration}")
            lateFusionInstance.pretty_print_performance_metric(baseClfPerfFlag=True, globlFusionPerfFlag=True, hpcGloblFusionPerfFlag=True)
        
        # Save the updated late stage fusion objects
        if saveLateStageFusionObject:
            savePath = os.path.join(self.xmd_base_folder_location, "res", "trainedModels", self.datasetName)
            if not os.path.isdir(savePath):
                os.mkdir(savePath)
            savePath = os.path.join(savePath,f"lateFusion_logRuntime{logcatRuntimeThreshold}_truncDuration{truncatedDuration}_malwarePercent{self.malwarePercent}_trainedModel{trainedModelDetails['logcatRuntimeThreshold']}_{trainedModelDetails['truncatedDuration']}_{trainedModelDetails['malwarePercent']}.pkl")
            self.lateStageFusionObject[logcatRuntimeThreshold][truncatedDuration] = savePath
            lateFusionInstance.save_fusion_object(fpath=savePath)

        return lateFusionInstance

    

    def logcat_runtime_vs_truncated_duration_grid(self, trainStage1, trainStage2, trainedModelDetails = None):
        """
        Performs a grid search over logcatRuntimeThreshold and truncatedDuration for the following tasks.
            Tasks:
                - For cd-dataset, generate evaluation scores using one of the following: 
                                (1) specified trained model [only when trainedModelDetails is passed], 
                                (2) trained model with the same config parameters: logcatRuntimeThreshold and truncatedDuration.
                - For std-dataset, generate late-stage-fusion instances storing the trained models.
            
            params:
                - trainStage1, trainStage2 (bool) : For std-dataset, if True, then will train the corresponding stage-1 and stage-2 classifiers.
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
                                            saveLateStageFusionObject=True)                
                    
        elif (datasetType == "std-dataset"):
            # Generate the trained models by doing gridsearch over logcatRuntimeThreshold and truncatedDuration.
            for logcatRuntimeThreshold in self.candidateLogcatRuntimeThresholds:
                for truncatedDuration in self.candidateTruncatedDurations:
                    orchInst = orchestrator(args=self.args, 
                                basePath_featureEngineeredDataset=self.basePath_featureEngineeredDataset, 
                                datasetName=self.datasetName, 
                                malwarePercent=self.malwarePercent,
                                xmd_base_folder_location=self.xmd_base_folder_location)
                    
                    orchInst.std_dataset_tasks(logcatRuntimeThreshold=logcatRuntimeThreshold, 
                                            truncatedDuration=truncatedDuration, 
                                            print_performance_metric=True, 
                                            saveTrainedModels=True,
                                            trainStage1=trainStage1, 
                                            trainStage2=trainStage2)                
        
        else:
            raise ValueError(f"Incomplete arguments : datasetType is {datasetType} and trainedModelDetails is {trainedModelDetails}.")


    def prettyPrintGridPerformanceMetrics(self, datasetName, performanceMetricTypeSpecifier, malwarePercent, trainedModelDetails=None):
        """
        Pretty prints the performance metrics over the search grid of logcatRuntimeThreshold and truncatedDuration.
        Edit this method if you want a different performance metric.
        
        params:
            - datasetName : Name of the dataset for which the performance metric grid needs to be printed [If dataset is cd, then check trainedModelDetails]
            - performanceMetricTypeSpecifier (dict): {
                                                        "performanceMetricDict": "stage1ClassifierPerformanceMetrics", "globlFusionPerformanceMetric", or "hpcGloblFusionPerformanceMetricAllGroup", 
                                                        "performanceMetricName": 'f1', 'precision', or 'recall', 
                                                        "selectedBaseClassifier": "globl", "hpc", or "all",
                                                        "globlChannelType": "globl", or "dvfs",
                                                        "hpc-group-name": "hpc-group-1", "hpc-group-2", "hpc-group-3", or "hpc-group-4",
                                                        "clfToi": "hpc", "hpc-dvfs-ensemble", "hpc-dvfs-sg-rf", "hpc-dvfs-sg-lr", "hpc-globl-ensemble", "hpc-globl-sg-rf", or "hpc-globl-sg-lr"
                                                    }
                                    NOTE:
                                        (1) if "performanceMetricDict": "stage1ClassifierPerformanceMetrics"
                                                Need: splitType, selectedBaseClassifier, performanceMetricName
                                        (2) if "performanceMetricDict": "globlFusionPerformanceMetric"
                                                Need: splitType, globlChannelType, performanceMetricName
                                        (3) if "performanceMetricDict": "hpcGloblFusionPerformanceMetricAllGroup"
                                                Need: splitType, hpc-group-name, clfToi, performanceMetricName

            - malwarePercent (int) : Used for accessing the corresponding late_stage_fusion instance
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
        clfToi_tracker = []

        datasetType, _ = dataloader_generator.get_dataset_type_and_partition_dist(dataset_name = datasetName)

        # For display
        gridTable = PrettyTable()
        gridTable.field_names = ["LogcatRuntimeThreshold"]+[truncatedDuration for truncatedDuration in self.candidateTruncatedDurations]

        for logcatRuntimeThreshold in self.candidateLogcatRuntimeThresholds:
            performanceMetricForAll_truncatedDuration = []

            for truncatedDuration in self.candidateTruncatedDurations:
                ################################### Load the saved lateFusionInstance ###################################
                savePath = os.path.join(self.xmd_base_folder_location, "res", "trainedModels", datasetName)
                
                if datasetType == "std-dataset":
                    savePath = os.path.join(savePath,f"lateFusion_logRuntime{logcatRuntimeThreshold}_truncDuration{truncatedDuration}_malwarePercent{malwarePercent}.pkl")
                elif datasetType == "cd-dataset" and (trainedModelDetails is not None): # Testing wrt the specified trained model
                    savePath = os.path.join(savePath,f"lateFusion_logRuntime{logcatRuntimeThreshold}_truncDuration{truncatedDuration}_malwarePercent{malwarePercent}_trainedModel{trainedModelDetails['logcatRuntimeThreshold']}_{trainedModelDetails['truncatedDuration']}_{trainedModelDetails['malwarePercent']}.pkl")
                elif datasetType == "cd-dataset": # Testing with the trained model of same config 
                    savePath = os.path.join(savePath,f"lateFusion_logRuntime{logcatRuntimeThreshold}_truncDuration{truncatedDuration}_malwarePercent{malwarePercent}_trainedModel{logcatRuntimeThreshold}_{truncatedDuration}_{malwarePercent}.pkl")
                else: 
                    raise ValueError(f"Incorrect dataset name specified: {datasetName}") 

                lateFusionInstance = late_stage_fusion(args=self.args)
                lateFusionInstance.load_fusion_object(fpath=savePath)
                #########################################################################################################

                #################################### Get the performance metric from the lateFusionInstance ###################################
                performanceMetricDict = performanceMetricTypeSpecifier["performanceMetricDict"]
                performanceMetricName = performanceMetricTypeSpecifier["performanceMetricName"]
                selectedBaseClassifier= performanceMetricTypeSpecifier["selectedBaseClassifier"]
                globlChannelType= performanceMetricTypeSpecifier["globlChannelType"]
                hpc_group_name= performanceMetricTypeSpecifier["hpc-group-name"]
                clfToi=performanceMetricTypeSpecifier["clfToi"]

                performanceMetricType = performanceMetricAggregator.generatePerformanceMetricTypeDict(datasetName=datasetName, 
                                                                                                        performanceMetricDict = performanceMetricDict, 
                                                                                                        performanceMetricName = performanceMetricName, 
                                                                                                        selectedBaseClassifier= selectedBaseClassifier, 
                                                                                                        globlChannelType=globlChannelType, 
                                                                                                        hpc_group_name=hpc_group_name, 
                                                                                                        clfToi=clfToi)

                pScore = performanceMetricAggregator.getAggregatePerformanceMetric(lateFusionInstance=lateFusionInstance, 
                                                                            performanceMetricType=performanceMetricType)
                pScore = round(pScore,2)
                performanceMetricForAll_truncatedDuration.append(pScore)

                # For the plots
                truncatedDuration_tracker.append(truncatedDuration)
                logcatRuntimeThreshold_tracker.append(logcatRuntimeThreshold)
                performanceMetric_tracker.append(pScore)
                clfToi_tracker.append(clfToi)
                ###############################################################################################################################

            gridTable.add_row([logcatRuntimeThreshold]+performanceMetricForAll_truncatedDuration)
            
        print(f" --- Performance metric for {datasetName} --- \nperformanceMetricDict: {performanceMetricDict}\nperformanceMetricName: {performanceMetricName}\
            \nselectedBaseClassifier: {selectedBaseClassifier}\ngloblChannelType: {globlChannelType}\nhpc_group_name: {hpc_group_name}\nclfToi: {clfToi}")
        print(gridTable)
        print("--------------------------------------------------")
        print(gridTable.get_csv_string())

        
        ########### Generate the plot ###########
        d = {'truncatedDuration':truncatedDuration_tracker, 'logcatRuntimeThreshold': logcatRuntimeThreshold_tracker, 'performanceMetric':performanceMetric_tracker}
        df = pd.DataFrame(data=d)

        if trainedModelDetails is not None:
            trainedModelFileName = f"trainedModel{trainedModelDetails['logcatRuntimeThreshold']}_{trainedModelDetails['truncatedDuration']}_{trainedModelDetails['malwarePercent']}"
        else:
            trainedModelFileName = None

        ImagePlottingTools.multiLinePlot(df=df, 
                                        performanceMetric=performanceMetricTypeSpecifier["performanceMetricName"], 
                                        plotTitle=f"Performance over lRT vs. tD grid {performanceMetricDict}",
                                        saveLocation=f"/data/hkumar64/projects/arm-telemetry/xmd/plots/analysis/{performanceMetricDict}_{performanceMetricName}_{selectedBaseClassifier}_{hpc_group_name}_{clfToi}_model_{trainedModelFileName}.png")
        #########################################

        return gridTable

    @staticmethod
    def print_lateFusionObject_performanceMetrics(xmd_base_folder_location, datasetName, truncatedDuration, logcatRuntimeThreshold, malwarePercent, trainedModelDetails, args):
        """
        Prints all the performance metric stored in the late fusion object
        params:
            - xmd_base_folder_location: Used for accessing the corresponding late stage fusion object
            - datasetName: Used for accessing the corresponding late stage fusion object
            - truncatedDuration, logcatRuntimeThreshold, malwarePercent: Used for accessing the late stage fusion object for the cd-dataset
            - trainedModelDetails: Used for accessing the late stage fusion object
        """
        # Load the late stage fusion object
        datasetType, _ = dataloader_generator.get_dataset_type_and_partition_dist(dataset_name = datasetName)
        if datasetType == 'std-dataset':
            savePath = os.path.join(xmd_base_folder_location, "res", "trainedModels", "std-dataset")
            savePath = os.path.join(savePath,f"lateFusion_logRuntime{trainedModelDetails['logcatRuntimeThreshold']}_truncDuration{trainedModelDetails['truncatedDuration']}_malwarePercent{trainedModelDetails['malwarePercent']}.pkl")
        elif datasetType == 'cd-dataset':
            savePath = os.path.join(xmd_base_folder_location, "res", "trainedModels", datasetName)
            savePath = os.path.join(savePath,f"lateFusion_logRuntime{logcatRuntimeThreshold}_truncDuration{truncatedDuration}_malwarePercent{malwarePercent}_trainedModel{trainedModelDetails['logcatRuntimeThreshold']}_{trainedModelDetails['truncatedDuration']}_{trainedModelDetails['malwarePercent']}.pkl")
        else:
            raise ValueError(f"Incorrect dataset name passed: {datasetName}")

        lateFusionInstance = late_stage_fusion(args=args)
        lateFusionInstance.load_fusion_object(fpath=savePath)

        # Pretty print the performance metrics
        lateFusionInstance.pretty_print_performance_metric(baseClfPerfFlag=True, globlFusionPerfFlag=True, hpcGloblFusionPerfFlag=True)

    def generate_performance_grid():
        """
        Generates the performance grid.

        params:
            - datasetName (str): Name of the dataset. Used for determining list of tasks.
            - createLateStageFusionObjects (bool): If True, then will create the late stage fusion object.
            - basePath_featureEngineeredDataset (str): Path of the base directory where the featureEngineered datasets are stored.
            - malwarePercent (float): [0.5 or 0.1] Percentage of malware in the dataset.
            - xmd_base_folder_location (str): Base location of the xmd directory
            - trainedModelDetails (dict): If passed and datasetType is cd, then will use the corresponding trained model for generating the grid.
            - performanceMetricTypeSpecifier (dict): {
                                                        "performanceMetricDict": "stage1ClassifierPerformanceMetrics", "globlFusionPerformanceMetric", or "hpcGloblFusionPerformanceMetricAllGroup", 
                                                        "performanceMetricName": 'f1', 'precision', or 'recall', 
                                                        "selectedBaseClassifier": "globl", "hpc", or "all",
                                                        "globlChannelType": "globl", or "dvfs",
                                                        "hpc-group-name": "hpc-group-1", "hpc-group-2", "hpc-group-3", or "hpc-group-4",
                                                        "clfToi": "hpc", "hpc-dvfs-ensemble", "hpc-dvfs-sg-rf", "hpc-dvfs-sg-lr", "hpc-globl-ensemble", "hpc-globl-sg-rf", or "hpc-globl-sg-lr"
                                                    }
                                    NOTE:
                                        (1) if "performanceMetricDict": "stage1ClassifierPerformanceMetrics"
                                                Need: splitType, selectedBaseClassifier, performanceMetricName
                                        (2) if "performanceMetricDict": "globlFusionPerformanceMetric"
                                                Need: splitType, globlChannelType, performanceMetricName
                                        (3) if "performanceMetricDict": "hpcGloblFusionPerformanceMetricAllGroup"
                                                Need: splitType, hpc-group-name, clfToi, performanceMetricName

        Output:

        """
        pass

    def save_orchestrator_state():
        pass

    def save_orchestrator_state():
        pass
        
    @staticmethod
    def unit_test_orchestrator(args, xmd_base_folder_location):
        basePath_featureEngineeredDataset="/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset1"
        # basePath_featureEngineeredDataset="/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDatasetWinter"
        ######################### Print the performance metrics of a late stage fusion object ##########################
        # datasetName = "cdyear1-dataset"
        # logcatRuntimeThreshold=0
        # truncatedDuration=15
        # malwarePercent=0.5
        # trainedModelDetails = {"logcatRuntimeThreshold":0, "truncatedDuration":15, "malwarePercent":0.5}

        # orchestrator.print_lateFusionObject_performanceMetrics(xmd_base_folder_location=xmd_base_folder_location, 
        #                                                         datasetName=datasetName, 
        #                                                         truncatedDuration=truncatedDuration, 
        #                                                         logcatRuntimeThreshold=logcatRuntimeThreshold, 
        #                                                         malwarePercent=malwarePercent, 
        #                                                         trainedModelDetails=trainedModelDetails, 
        #                                                         args=args)
        # exit()
        # ###############################################################################################################
        ########################## Testing std-dataset tasks ##########################
        # orchInst = orchestrator(args=args, 
        #                         basePath_featureEngineeredDataset=basePath_featureEngineeredDataset, 
        #                         datasetName="std-dataset", 
        #                         malwarePercent=0.5,
        #                         xmd_base_folder_location=xmd_base_folder_location)
        # orchInst.std_dataset_tasks(logcatRuntimeThreshold=15, truncatedDuration=30, print_performance_metric=True, saveTrainedModels=True, trainStage1=True, trainStage2=True)                
        # exit()
        ##############################################################################

        ########################## Testing cd-dataset tasks ###########################
        orchInst = orchestrator(args=args, 
                                basePath_featureEngineeredDataset=basePath_featureEngineeredDataset, 
                                datasetName="cd-dataset", 
                                malwarePercent=0.5,
                                xmd_base_folder_location=xmd_base_folder_location)
        # Trained model to be used for testing
        trainedModelDetails = {"logcatRuntimeThreshold":15, "truncatedDuration":30, "malwarePercent":0.5}
        orchInst.cd_dataset_tasks(trainedModelDetails=trainedModelDetails, logcatRuntimeThreshold=15, truncatedDuration=30, print_performance_metric=True, saveLateStageFusionObject=True)                
        exit()
        # ##############################################################################

        ######################## Testing grid search task ##########################
        # orchInst = orchestrator(args=args, 
        #                         basePath_featureEngineeredDataset="/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDatasetWinter", 
        #                         datasetName="cdyear3-dataset", 
        #                         malwarePercent=0.5,
        #                         xmd_base_folder_location=xmd_base_folder_location)
        # trainedModelDetails = {"logcatRuntimeThreshold":75, "truncatedDuration":90, "malwarePercent":0.5}
        # # trainedModelDetails=None
        # orchInst.logcat_runtime_vs_truncated_duration_grid(trainStage1=True,
        #                                                     trainStage2=True, 
        #                                                     trainedModelDetails=trainedModelDetails) 
        # exit()
        #############################################################################

        # ########################## Testing performance metric aggregator ########################
        ## To be used as reference for passing arguments
        # "performanceMetricDict": "stage1ClassifierPerformanceMetrics", "globlFusionPerformanceMetric", or "hpcGloblFusionPerformanceMetricAllGroup", 
        #                                                 "performanceMetricName": 'f1', 'precision', or 'recall', 
        #                                                 "selectedBaseClassifier": "globl", "hpc", or "all",
        #                                                 "globlChannelType": "globl", or "dvfs",
        #                                                 "hpc-group-name": "hpc-group-1", "hpc-group-2", "hpc-group-3", or "hpc-group-4",
        #                                                 "clfToi": "hpc", "hpc-dvfs-ensemble", "hpc-dvfs-sg-rf", "hpc-dvfs-sg-lr", "hpc-globl-ensemble", "hpc-globl-sg-rf", or "hpc-globl-sg-lr"
        
        # datasetName = "cdyear1-dataset"
        # performanceMetricDict = "hpcGloblFusionPerformanceMetricAllGroup"
        # performanceMetricName = 'f1'
        # selectedBaseClassifier= "all"
        # globlChannelType= "dvfs"
        # hpc_group_name= "hpc-group-4"
        # clfToi="hpc-globl-ensemble"
        
        # performanceMetricType =performanceMetricAggregator.generatePerformanceMetricTypeDict(datasetName=datasetName, 
        #                                                                 performanceMetricDict = performanceMetricDict, 
        #                                                                 performanceMetricName = performanceMetricName, 
        #                                                                 selectedBaseClassifier= selectedBaseClassifier, 
        #                                                                 globlChannelType=globlChannelType, 
        #                                                                 hpc_group_name=hpc_group_name, 
        #                                                                 clfToi=clfToi)

        # # Load the saved lateFusionInstance 
        # savePath = os.path.join(xmd_base_folder_location, "res", "trainedModels", datasetName)
        # savePath = os.path.join(savePath,f"lateFusion_logRuntime{0}_truncDuration{30}.pkl")
        # lateFusionInstance = late_stage_fusion(args=args)
        # lateFusionInstance.load_fusion_object(fpath=savePath)
        # lateFusionInstance.pretty_print_performance_metric(baseClfPerfFlag=True, globlFusionPerfFlag=True, hpcGloblFusionPerfFlag=True)

        # pScore = performanceMetricAggregator.getAggregatePerformanceMetric(lateFusionInstance=lateFusionInstance, 
        #                                                                     performanceMetricType=performanceMetricType)
        # print(pScore)
        # #########################################################################################

        ########################## Testing the performance grid generator ##########################
        orchInst = orchestrator(args=args, 
                                basePath_featureEngineeredDataset="/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDatasetWinter", 
                                datasetName="cdyear2-dataset", 
                                malwarePercent=0.5,
                                xmd_base_folder_location=xmd_base_folder_location)

        ############################################################## To be used as reference ###################################################################################################
        # performanceMetricTypeSpecifier = {    
        #                                     "performanceMetricDict": "stage1ClassifierPerformanceMetrics", "globlFusionPerformanceMetric", or "hpcGloblFusionPerformanceMetricAllGroup", 
        #                                     "performanceMetricName": 'f1', 'precision', or 'recall', 
        #                                     "selectedBaseClassifier": "globl", "hpc", or "all",
        #                                     "globlChannelType": "globl", or "dvfs",
        #                                     "hpc-group-name": "hpc-group-1", "hpc-group-2", "hpc-group-3", or "hpc-group-4",
        #                                     "clfToi": "hpc", "hpc-dvfs-ensemble", "hpc-dvfs-sg-rf", "hpc-dvfs-sg-lr", "hpc-globl-ensemble", "hpc-globl-sg-rf", or "hpc-globl-sg-lr"
        #                                 }
        ##########################################################################################################################################################################################

        performanceMetricTypeSpecifier = {
                                            "performanceMetricDict": "hpcGloblFusionPerformanceMetricAllGroup", 
                                            "performanceMetricName": 'f1', 
                                            "selectedBaseClassifier": "hpc",
                                            
                                            "globlChannelType": "globl",
                                            
                                            "hpc-group-name": "hpc-group-4",
                                            "clfToi": "hpc-dvfs-sg-rf"
                                        }
        trainedModelDetails = {"logcatRuntimeThreshold":0, "truncatedDuration":90, "malwarePercent":0.5}
        orchInst.prettyPrintGridPerformanceMetrics(datasetName = "cdyear2-dataset", 
                                                    performanceMetricTypeSpecifier = performanceMetricTypeSpecifier,
                                                    malwarePercent=0.5,
                                                    trainedModelDetails=trainedModelDetails)
        ############################################################################################





def main_worker(args, xmd_base_folder_location):
    """
    Worker node that performs the complete analysis.
    params:
        - args: easydict storing the experimental parameters
        - xmd_base_folder_location: Location of base folder of xmd
    """
    # resample_dataset.unitTestResampler()
    baseRFmodel.unit_test_baseRFmodel(args=args)
    # late_stage_fusion.unit_test_lateStageFusion(args=args)
    # featureEngineeredDatasetLoader.unit_test_featureEngineeredDatasetLoader()
    # orchestrator.unit_test_orchestrator(args=args, xmd_base_folder_location= xmd_base_folder_location)


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