from sales.exception.exception import SalesException
from sales.logger.logging import logging
import pandas as pd
import numpy as np
import os,sys
from sales.entity.config_entity import ModelTrainerConfig
from sales.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from sales.entity.model_factory import ModelFactory,MetricInfoArtifact,evaluate_regression_model
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
            raise SalesException(e,sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"

class ModelTrainer:

    def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_confg:ModelTrainerConfig):
        try:
            logging.info("..............Model-Training-Log-Started...................")
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_config=model_trainer_confg
        except Exception as e:
            raise SalesException(e,sys) from e

    def initiate_model_training(self)->ModelTrainerArtifact:
        try:
            os.makedirs(self.model_trainer_config.trained_model_dir,exist_ok=True)
            os.makedirs(self.model_trainer_config.model_config_dir,exist_ok=True)
            trained_model_object_file_path=os.path.join(self.model_trainer_config.trained_model_dir,self.model_trainer_config.model_file_name)
            preprocessing_obj_file_path=os.path.join(self.data_transformation_artifact.preprocessed_object_file_path)
            model_config_file_file_path=os.path.join(self.model_trainer_config.model_config_dir,self.model_trainer_config.model_config_file_name)
            train_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_dir_path)
            test_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_dir_path)

            logging.info("Splitting the transformed train and test array")
            train_input_arr=train_arr[:,:-1]
            test_input_arr=test_arr[:,:-1]
            train_output_arr=train_arr[:,-1]
            test_output_arr=test_arr[:,-1]
            logging.info("Initializing ModelFactory class by specifyig Model Configuration path")
            model_factory=ModelFactory(model_config_path=model_config_file_file_path)
            
            base_accuracy=self.model_trainer_config.base_accuracy
            logging.info(f"Expected accuracy:{base_accuracy}")

            logging.info("Getting best model on the training dataset")
            best_model=model_factory.get_best_model(X=train_input_arr,y=train_output_arr,base_accuracy=base_accuracy)
            logging.info(f"Best model on training dataset:{best_model}")

            logging.info("Extracting trained model list")
            grid_searched_best_model_list=model_factory.grid_searched_best_model_list

            model_list=[model.best_model for model in grid_searched_best_model_list]
            logging.info("Evaluating grid searched best model on training as well as testing dataset")

            metric_info_artifact=evaluate_regression_model(model_list=model_list,X_train=train_input_arr,y_train=train_output_arr,X_test=test_input_arr,y_test=test_output_arr,base_accuracy=base_accuracy)
            logging.info(f"Best model metric info data:{metric_info_artifact}")

            preprocessing_obj=loading_object(file_path=preprocessing_obj_file_path)
            model_trainer_object=metric_info_artifact.model_object

            sales_model=SalesPredictor(preprocessing_object=preprocessing_obj,trained_model_object=model_trainer_object)
            logging.info(f"Saving model at file_path:{trained_model_object_file_path}")
            saving_object(file_path=trained_model_object_file_path,obj=sales_model)

            model_trainer_artifact=ModelTrainerArtifact(
                trained_model_file_path=trained_model_object_file_path, 
                is_trained=True, 
                message="Model Training Done Successfully", 
                train_rmse=metric_info_artifact.train_rmse, 
                test_rmse=metric_info_artifact.test_rmse, 
                train_accuracy=metric_info_artifact.train_accuracy, 
                test_accuracy=metric_info_artifact.test_accuracy, 
                model_accuracy=metric_info_artifact.model_accuracy
            )
            logging.info(f"Model Trainer Artifact:{model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise SalesException(e,sys) from e

    def __del__(self):
        logging.info("..............Model-Training-Log-Completed...................")
        

