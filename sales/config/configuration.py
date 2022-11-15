import os,sys
from sales.logger.logging import logging
from sales.exception.exception import SalesException
from sales.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig,DataTransformationConfig,ModelEvaluationConfig,ModelTrainerConfig,ModelPusherConfig
from sales.constant.constants import *
from sales.helper_functions.helper import read_yaml_file

class Configuration:

    def __init__(self,
        config_file_path:str=CONFIG_FILE_PATH,
        current_time_stamp:str=CURRENT_TIME_STAMP
        ) -> None:
        try:
            self.config_info=read_yaml_file(file_path=config_file_path)
            self.training_pipeline_config = self.get_training_pipeline_config()
            self.time_stamp=current_time_stamp
        except Exception as e:
            raise SalesException(e,sys) from e

    def get_data_ingestion_config(self)-> DataIngestionConfig:
        try:
            data_ingestion_info=self.config_info[DATA_INGESTION_CONFIG_KEY]
            data_ingestion_artifact_dir=os.path.join(
                self.training_pipeline_config.artifact_dir,
                DATA_INGESTION_ARTIFACT_DIR,
                self.time_stamp
                )
            dataset_dir=os.path.join(ROOT_DIR,data_ingestion_info[DATA_INGESTION_DATA_DIR])    
            data_ingestion_dir=os.path.join(data_ingestion_artifact_dir,data_ingestion_info[DATA_INGESTION_INGESTED_DIR])
            data_ingestion_train_dir=os.path.join(data_ingestion_dir,data_ingestion_info[DATA_INGESTION_INGESTED_TRAIN_DIR])
            data_ingestion_test_dir=os.path.join(data_ingestion_dir,data_ingestion_info[DATA_INGESTION_INGESTED_TEST_DIR])
            data_ingestion_config=DataIngestionConfig(
                data_ingestion_dir=data_ingestion_dir,
                ingested_train_dir=data_ingestion_train_dir,
                ingested_test_dir=data_ingestion_test_dir,
                data_dir=dataset_dir,
                train_file_name=data_ingestion_info[DATA_INGESTION_TRAIN_NAME_KEY],
                test_file_name=data_ingestion_info[DATA_INGESTION_TEST_NAME_KEY]
            )
            return data_ingestion_config
        except Exception as e:
            raise SalesException(e,sys) from e

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            data_validation_info=self.config_info[DATA_VALIDATION_CONFIG_KEY]
            data_validation_artifact_dir=os.path.join(
                self.training_pipeline_config.artifact_dir,
                DATA_VALIDATION_ARTIFACT_DIR,
                self.time_stamp
                )
            schema_dir=os.path.join(ROOT_DIR,data_validation_info[DATA_VALIDATION_SCHEMA_DIR])      
            data_validation_config=DataValidationConfig(
                schema_dir=schema_dir, 
                data_validation_dir=data_validation_artifact_dir, 
                schema_file_name=data_validation_info[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY], 
                report_file_name=data_validation_info[DATA_VALIDATION_REPORT_FILE_NAME], 
                report_page_file_name=data_validation_info[DATA_VALIDATION_REPORT_PAGE_FILE_NAME]
            )
            return data_validation_config
        except Exception as e:
            raise SalesException(e,sys) from e

    def get_data_tranformation_config(self)->DataTransformationConfig:
        try:
            data_transformation_info=self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]
            data_transformation_artifact_dir=os.path.join(
                self.training_pipeline_config.artifact_dir,
                DATA_TRANSFORMATION_ARTIFACT_DIR,
                self.time_stamp
            )
            transformed_train_dir=os.path.join(data_transformation_artifact_dir,data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_DIR],data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR])
            transformed_test_dir=os.path.join(data_transformation_artifact_dir,data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_DIR],data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR])
            preprocessing_dir=os.path.join(data_transformation_artifact_dir,data_transformation_info[DATA_TRANSFORMATION_PREPROCESSING_DIR])
            data_transformation_config=DataTransformationConfig(
                transformed_train_dir=transformed_train_dir,
                transformed_test_dir=transformed_test_dir,
                preprocessing_dir=preprocessing_dir,
                preprocessing_object_file_name=data_transformation_info[DATA_TRANSFORMATION_PREPROCESSING_FILE_NAME_KEY]
            )
            return data_transformation_config
        except Exception as e:
            raise SalesException(e,sys) from e

    def get_model_trainer_config(self)->ModelTrainerConfig:
        try:
            model_training_info=self.config_info[MODEL_TRAINER_CONFIG_KEY]
            model_training_artifact_dir=os.path.join(
                self.training_pipeline_config.artifact_dir,
                MODEL_TRAINER_ARTIFACT_DIR,
                self.time_stamp
            )
            model_config_dir=os.path.join(ROOT_DIR,model_training_info[MODEL_TRAINER_MODEL_CONFIG_DIR])
            trained_model_dir=os.path.join(model_training_artifact_dir,model_training_info[MODEL_TRAINER_TRAINED_MODEL_DIR])
            model_trainer_config=ModelTrainerConfig(
                trained_model_dir=trained_model_dir, 
                model_file_name=model_training_info[MODEL_TRAINER_MODEL_FILE_NAME_KEY], 
                base_accuracy=model_training_info[MODEL_TRAINER_BASE_ACCURACY_KEY], 
                model_config_dir=model_config_dir, 
                model_config_file_name=model_training_info[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY]
            )
            return model_trainer_config
        except Exception as e:
            raise SalesException(e,sys) from e

    def get_model_evaluation_config(self)->ModelEvaluationConfig:
        try:
            model_evaluation_config=self.config_info[MODEL_EVALUATION_CONFIG_KEY]
            model_evaluation_dir=os.path.join(
                self.training_pipeline_config.artifact_dir,
                MODEL_EVALUATION_ARTIFACT_DIR
            )
            model_evaluation_config=ModelEvaluationConfig(
                model_evaluation_dir=model_evaluation_dir, 
                model_evaluation_file_name=model_evaluation_config[MODEL_EVALUATION_FILE_NAME], 
                time_stamp=self.time_stamp
            )
            return model_evaluation_config
        except Exception as e:
            raise SalesException(e,sys) from e

    def get_model_pusher_config(self)->ModelPusherConfig:
        try:
            time_stamp=get_current_time_stamp()
            model_pusher_info=self.config_info[MODEL_PUSHER_CONFIG_KEY]
            model_push_dir=os.path.join(ROOT_DIR,
                model_pusher_info[MODEL_PUSHER_MODEL_EXPORT_DIR],
                time_stamp
                )
            model_push_config=ModelPusherConfig(
                model_export_dir=model_push_dir
            ) 
            return model_push_config   
        except Exception as e:
            raise SalesException(e,sys) from e
                                    
    def get_training_pipeline_config(self)->TrainingPipelineConfig:
        try:
            training_pipeline_info=self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            training_pipeline_artifact_dir=os.path.join(ROOT_DIR,training_pipeline_info[TRAINING_PIPELINE_NAME],training_pipeline_info[TRAINING_PIPELINE_ARTIFACT_DIR])
            training_pipeline_name=training_pipeline_info[TRAINING_PIPELINE_NAME]
            training_pipeline_config=TrainingPipelineConfig(
                pipeline_name=training_pipeline_name,
                artifact_dir=training_pipeline_artifact_dir
            )
            return training_pipeline_config
        except Exception as e:
            raise SalesException(e,sys) from e 
