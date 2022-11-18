from sales.exception.exception import SalesException
from sales.logger.logging import logging
import pandas as pd
import numpy as np
import os,sys
from sales.entity.config_entity import ModelTrainerConfig
from sales.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,ModelTrainerArtifact
from sales.helper_functions.helper import *
from sales.constant.constants import *
from typing import List

class SalesPredictor:

    def __init__(self,preprocessing_object,trained_model_object):
        """
        We can use this class for prediction on the test dataset by combining 
        one,the preprocessing step and other is the trained model object prediction step
        """
        self.preprocessing_object=preprocessing_object
        self.trained_model_object=trained_model_object

    def predict(self,X)->np.ndarray:
        try:
            transformed_feature=self.preprocessing_object.transform(X)
            return self.trained_model_object.predict(transformed_feature) 
        except Exception as e:
            raise SalesException from e

class ModelTrainer:

    def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_confg:ModelTrainerConfig):
        try:
            logging.info("..............Model-Training-Log-Started...................")
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_config=model_trainer_confg
        except Exception as e:
            raise SalesException(e,sys) from e

    def initiate_model_training(self):
        try:
            pass
        except Exception as e:
            raise SalesException(e,sys) from e

    def __del__(self):
        logging.info("..............Model-Training-Log-Completed...................")
        

