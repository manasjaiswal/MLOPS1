from sales.exception.exception import SalesException
from sales.logger.logging import logging
import pandas as pd
import numpy as np
import os,sys
from sales.entity.config_entity import ModelEvaluationConfig
from sales.entity.artifact_entity import DataIngestionArtifact,ModelTrainerArtifact,ModelEvaluationArtifact,DataValidationArtifact
from sales.entity.model_factory import evaluate_regression_model,MetricInfoArtifact
from sales.helper_functions.helper import *
from sales.constant.constants import *
from typing import List

class ModelEvaluation:

    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,model_evaluation_config:ModelEvaluationConfig,model_trainer_artifact:ModelTrainerArtifact,data_validation_artifact:DataValidationArtifact):
        try:
            logging.info("..............Model-Evaluation-Log-Started............")
            self.model_evaluation_config=model_evaluation_config
            self.model_trainer_artifact=model_trainer_artifact
            self.data_validation_artifact=data_validation_artifact
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise SalesException(e,sys) from e

    def get_best_model(self):
        try:
            model=None
            model_evaluation_file_path=os.path.join(self.model_evaluation_config.model_evaluation_dir,self.model_evaluation_config.model_evaluation_file_name)                             
            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path)
                return model
            model_eval_file_content=read_yaml_file(file_path=model_evaluation_file_path)

            model_eval_file_content=dict() if model_eval_file_content is None else model_eval_file_content      

            if BEST_MODEL_KEY not in model_eval_file_content:
                return model

            model=loading_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model    
        except Exception as e:
            raise SalesException(e,sys) from e

    def update_evaluation_report(self,model_evaluation_artifact:ModelEvaluationArtifact):
        try:
            eval_file_path=os.path.join(self.model_evaluation_config.model_evaluation_dir,self.model_evaluation_config.model_evaluation_file_name)
            model_eval_content=read_yaml_file(file_path=eval_file_path)
            model_eval_content=dict() if model_eval_content is None else model_eval_content

            previous_best_model=None
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model=model_eval_content[BEST_MODEL_KEY]
            logging.info(f"Previous best model:{previous_best_model}")
            eval_result={
                BEST_MODEL_KEY:{
                    MODEL_PATH_KEY:model_evaluation_artifact.evaluated_model_path
                }
            }    
            if previous_best_model is not None:
                model_history={self.model_evaluation_config.time_stamp:previous_best_model}
                if HISTORY_KEY not in model_eval_content:
                    history={HISTORY_KEY:model_history}
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)

            model_eval_content.update(eval_result)
            logging.info(f"Updated eval result:{model_eval_content}")
            write_yaml_file(file_path=eval_file_path,json_content=model_eval_content)            
        except Exception as e:
            raise SalesException(e,sys) from e

    def initiate_model_evaluation(self):
        try:
            trained_model_file_path=self.model_trainer_artifact.trained_model_file_path
            trained_model_object=loading_object(file_path=trained_model_file_path)

            train_file_path=self.data_ingestion_artifact.train_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path

            schema_file_path=self.data_validation_artifact.schema_file_path 

            train_data_frame=load_data(file_path=train_file_path,schema_file_path=schema_file_path)
            test_data_frame=load_data(file_path=test_file_path,schema_file_path=schema_file_path)   

            schema_content=read_yaml_file(file_path=schema_file_path)
            target_column_name=schema_content[SCHEMA_CONFIG_TARGET_COLUMN_KEY]

            X_train=train_data_frame.drop(columns=[target_column_name])
            y_train=train_data_frame[target_column_name]
            X_test=test_data_frame.drop(columns=[target_column_name])
            y_test=test_data_frame[target_column_name]

            model=self.get_best_model()

            if model is None:
                logging.info("Not found any existing model,hence aacepting the trained model")
                model_evaluation_artifact=ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,is_model_accepted=True)
                
                self.update_evaluation_report(model_evaluation_artifact=model_evaluation_artifact)
                logging.info(f"Model accepted Model evaluation artifact{model_evaluation_artifact} created")
                return model_evaluation_artifact

            model_list=[model,trained_model_object]

            metric_info_artifact:MetricInfoArtifact=evaluate_regression_model(model_list=model_list,
                                                                            X_train=X_train,
                                                                            X_test=X_test,
                                                                            y_train=y_train,
                                                                            y_test=y_test,
                                                                            base_accuracy=self.model_trainer_artifact.model_accuracy
                                                                            )
            logging.info(f"Model evaluation completed. model metric artifact:{metric_info_artifact}")

            if metric_info_artifact is None:
                response=ModelEvaluationArtifact(is_model_accepted=False,evaluated_model_path=trained_model_file_path)
                logging.info(response)
            if metric_info_artifact.index_number==1:
                model_evaluation_artifact=ModelEvaluationArtifact(is_model_accepted=True,evaluated_model_path=trained_model_file_path)
                self.update_evaluation_report(model_evaluation_artifact=model_evaluation_artifact) 
                logging.info(f"model evaluation artifact:{model_evaluation_artifact}")
            else:
                model_evaluation_artifact=ModelEvaluationArtifact(is_model_accepted=False,evaluated_model_path=trained_model_file_path)
                logging.info(f"Trained model is not better than the existing model")

            return model_evaluation_artifact        
        except Exception as e:
            raise SalesException(e,sys) from e

    def __del__(self):
        logging.info("..........Model-Evaluation-Log-completed...............")


