from collections import namedtuple

#Dataingestion Configuration structure
DataIngestionConfig=namedtuple('DataIngestionConfig',["data_ingestion_dir","ingested_train_dir","ingested_test_dir","data_dir","train_file_name","test_file_name"])

#Training Pipeline Configuration structure
TrainingPipelineConfig=namedtuple('TrainingPipelineConfig',["pipeline_name","artifact_dir"])

#DataValidation Configuration structure
DataValidationConfig=namedtuple('DataValidationConfig',["schema_dir","data_validation_dir","schema_file_name","report_file_name","report_page_file_name"])

#DataTransformation Configuration structure
DataTransformationConfig=namedtuple('DataTranformationConfig',["transformed_train_dir","transformed_test_dir","preprocessing_dir","preprocessing_object_file_name"])

#ModelTrainer Configuration structure
ModelTrainerConfig=namedtuple('ModelTrainerConfig',["trained_model_dir","model_file_name","base_accuracy","model_config_dir","model_config_file_name"])

#ModelEvaluation Configuration structure
ModelEvaluationConfig=namedtuple("ModelEvaluationConfig",["model_evaluation_dir","model_evaluation_file_name","time_stamp"])

#ModelPusher Configuration structure
ModelPusherConfig=namedtuple('ModelPusherConfig',["model_export_dir"])