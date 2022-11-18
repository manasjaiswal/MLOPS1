from sales.exception.exception import SalesException
from sales.logger.logging import logging
from sales.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from sales.entity.config_entity import DataValidationConfig
import os,sys
import pandas as pd
import numpy as np
from sales.helper_functions.helper import read_yaml_file
from sales.constant.constants import *
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import json

class DataValidation:

    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        try:
            logging.info(".............DataValidation-Log-started............")
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self.schema_info=read_yaml_file(file_path=os.path.join(self.data_validation_config.schema_dir,self.data_validation_config.schema_file_name))
        except Exception as e:
            raise SalesException(e,sys) from e
    
    def validation_existence_of_files(self):
        try:
            train_existence=False
            test_existence=False
            prediction_file_existence=False
            if os.path.exists(self.data_ingestion_artifact.train_file_path):
                logging.info(f"Train file exists")
                train_existence=True
            else:
                logging.info("Trainfile missing")    
            if os.path.exists(self.data_ingestion_artifact.test_file_path):
                logging.info(f"Test file exists")
                test_existence=True        
            else:
                logging.info("Testfile missing")
            if os.path.exists(self.data_ingestion_artifact.prediction_file_path):
                logging.info(f"Prediction file exists")
                prediction_file_existence=True     
            if train_existence==False or test_existence==False or prediction_file_existence==False:
                raise SalesException("Files missing",sys)                    
        except Exception as e:
            raise SalesException(e,sys) from e
    
    def validation_of_schema(self):
        try:
            """
            we will check schema of train_file_path and prediction_file_path
            only,because schema of train and test wold be same because of splitting
            """
            logging.info("Checking schema of the datasets")
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            train_df=train_df.fillna(0)
            prediction_df=pd.read_csv(self.data_ingestion_artifact.prediction_file_path)
            prediction_df=prediction_df.fillna(0)
            schema_numerical_list=self.schema_info[SCHEMA_CONFIG_NUMERICAL_COLUMNS_KEY]
            schema_categorical_list=self.schema_info[SCHEMA_CONFIG_CATEGORICAL_COLUMNS_KEY]
            schema_target_column=self.schema_info[SCHEMA_CONFIG_TARGET_COLUMN_KEY]
            logging.info("Cheching DataTypes of columns of training and testing dataset")
            for feature in schema_numerical_list+schema_categorical_list+[schema_target_column]:
                if train_df[feature].dtype==self.schema_info[SCHEMA_CONFIG_DATA_TYPES_KEY][feature]:
                    pass
                else:
                    raise SalesException(f"Column name={feature}in dataset doesnot match with schema",sys)
            for feature in schema_numerical_list+schema_categorical_list:
                if prediction_df[feature].dtype==self.schema_info[SCHEMA_CONFIG_DATA_TYPES_KEY][feature]:
                    pass
                else:
                    raise SalesException(f"Column name={feature}in prediction dataset doesnot match with schema",sys)
            logging.info(f"Both datasets have same datatypes as that of the schema")
            logging.info(f"Checking for categories of categorical column")
            for feature in schema_categorical_list:
                #Filling Null values with 0 in categorical columns for this purpose only
                train_feature_list=list(train_df[feature].unique())
                prediction_feature_list=list(prediction_df[feature].unique())
                schema_feature_list=self.schema_info[SCHEMA_CONFIG_CATEGORIES_KEY][feature]
                for i in train_feature_list:
                    if i!=0:
                        if i in schema_feature_list:
                            pass
                        else:
                            raise Exception(f"category {i} of column{feature} in dataset doesnot lie in schema")
                for j in prediction_feature_list:
                    if j!=0:
                        if j in schema_feature_list:
                            pass
                        else:
                            raise Exception(f"category {i} of column{feature} in prediction dataset doesnot lie in schema")        
            logging.info("Both datasets have categories same as present in schema")    
        except Exception as e:
            raise SalesException(e,sys) from e

    def saving_data_drift_report(self)->dict:
        """
        We will check the data drift in training and prediction dataset
        because drift in train and test will always be negligible
        """
        try:
            profile=Profile(sections=[DataDriftProfileSection()])
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            prediction_df=pd.read_csv(self.data_ingestion_artifact.prediction_file_path)
            #We have to drop target column in train because it is not present in prediction_df
            train_df.drop(columns=self.schema_info[SCHEMA_CONFIG_TARGET_COLUMN_KEY],inplace=True)

            profile.calculate(train_df,prediction_df)

            report=json.loads(profile.json())

            report_file_path=os.path.join(self.data_validation_config.data_validation_dir,self.data_validation_config.report_file_name)
            os.makedirs(self.data_validation_config.data_validation_dir,exist_ok=True)

            with open(report_file_path,"w") as report_file:
                json.dump(report,report_file,indent=6)

            logging.info("report file in json saved")
            return report    
        except Exception as e:
            raise SalesException(e,sys) from e

    def saving_data_drift_report_page(self):
        try:
            dashboard=Dashboard(tabs=[DataDriftTab()])
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            prediction_df=pd.read_csv(self.data_ingestion_artifact.prediction_file_path)
            #We have to drop target column in train because it is not present in prediction_df
            train_df.drop(columns=self.schema_info[SCHEMA_CONFIG_TARGET_COLUMN_KEY],inplace=True)

            dashboard.calculate(train_df,prediction_df)

            report_page_file_path=os.path.join(self.data_validation_config.data_validation_dir,self.data_validation_config.report_page_file_name)
            
            dashboard.save(report_page_file_path)
            logging.info("report page file in html format saved")
        except Exception as e:
            raise SalesException(e,sys) from e 

    def is_data_drift_found(self)->bool:
        try:
            report=self.saving_data_drift_report()
            self.saving_data_drift_report_page()
            return True
        except Exception as e:
            raise SalesException(e,sys) from e 

    def validation_of_nan_values(self)->bool:
        """
        Nan Values have to be checked for all the three datasets we have
        trining,testing and prediction based dataset
        """
        try:
            null_existence=False
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            prediction_df=pd.read_csv(self.data_ingestion_artifact.prediction_file_path)
            #Avoiding Target Column because we cannot do something to nan values in target column
            train_df.drop(columns=self.schema_info[SCHEMA_CONFIG_TARGET_COLUMN_KEY],inplace=True)
            test_df.drop(columns=self.schema_info[SCHEMA_CONFIG_TARGET_COLUMN_KEY],inplace=True)
            numerical_column_list=self.schema_info[SCHEMA_CONFIG_NUMERICAL_COLUMNS_KEY]
            categorical_column_list=self.schema_info[SCHEMA_CONFIG_CATEGORICAL_COLUMNS_KEY]
            for feature in numerical_column_list+categorical_column_list:
                if True in train_df[feature].isnull().unique():
                    logging.info(f"Null values detected in column {feature} of training dataset")
                    null_existence=True
                if True in test_df[feature].isnull().unique():
                    logging.info(f"Null values detected in column {feature} of testing dataset")    
                    null_existence=True
                if True in prediction_df[feature].isnull().unique():
                    logging.info(f"Null values detected in column {feature} of prediction dataset")
                    null_existence=True
            return null_existence        
        except Exception as e:
            raise SalesException(e,sys) from e         
    
    def validation_of_outliers_values(self)->bool:
        """
        Checking outliers in all the numerical columns of the training and prediction dataset
        """
        try:
            outlier_existence=False
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            prediction_df=pd.read_csv(self.data_ingestion_artifact.prediction_file_path)
            #Avoiding Target Column because we cannot do something to outlier values in target column
            train_df.drop(columns=self.schema_info[SCHEMA_CONFIG_TARGET_COLUMN_KEY],inplace=True)
            numerical_column_list=self.schema_info[SCHEMA_CONFIG_NUMERICAL_COLUMNS_KEY]
            for feature in numerical_column_list:
                outliers1=[]
                arr1=np.percentile(train_df[feature],[25,75])
                iqr1=arr1[1]-arr1[0]
                lf1=arr1[0]-1.5*iqr1
                uf1=arr1[1]+1.5*iqr1
                for i in train_df[feature]:
                    if  lf1<i<uf1:
                        pass
                    else:
                        outliers1.append(i)
                if len(outliers1)>=1:
                    logging.info(f"Outliers detected in {feature} of training dataset")
                    outlier_existence=True
                outliers2=[]
                arr2=np.percentile(prediction_df[feature],[25,75])
                iqr2=arr2[1]-arr2[0]
                lf2=arr2[0]-1.5*iqr2
                uf2=arr2[1]+1.5*iqr2
                for j in prediction_df[feature]:
                    if  lf2<j<uf2:
                        pass
                    else:
                        outliers2.append(j)
                if len(outliers2)>=1:
                    logging.info(f"Outliers detected in {feature} of prediction based dataset")
                    outlier_existence=True
            return outlier_existence
        except Exception as e:
            raise SalesException(e,sys) from e

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            self.validation_existence_of_files()
            self.validation_of_schema()
            nan_existence=self.validation_of_nan_values()
            outlier_existence=self.validation_of_outliers_values()
            data_drift_existence=self.is_data_drift_found()
            data_validation_artifact=DataValidationArtifact(
                is_validated=True,
                report_file_path=os.path.join(self.data_validation_config.data_validation_dir,self.data_validation_config.report_file_name),
                report_page_file_path=os.path.join(self.data_validation_config.data_validation_dir,self.data_validation_config.report_page_file_name),
                null_existence=nan_existence, 
                outlier_existence=outlier_existence, 
                data_drift_existence=data_drift_existence,
                schema_file_path=os.path.join(self.data_validation_config.schema_dir,self.data_validation_config.schema_file_name)
            )
            return data_validation_artifact
        except Exception as e:
            raise SalesException(e,sys) from e        

    def __del__(self):
        logging.info(".........DataValidationLogCompleted..............")
