from collections import namedtuple


#DataIngestionArtifact

DataIngestionArtifact=namedtuple("DataIngestionArtifact",["train_file_path","test_file_path","prediction_file_path","is_ingested","message"])

#DataValidationArtifact

DataValidationArtifact=namedtuple("DataValidationArtifact",["schema_file_path","report_file_path","report_page_file_path","is_validated","null_existence","outlier_existence","data_drift_existence"])

#DataTransformationArtifact

DataTransformationArtifact=namedtuple("DataTransformationArtifact",["transformed_train_dir_path","transformed_test_dir_path","preprocessed_object_file_path","message"])

#Model Trainer Artifact
ModelTrainerArtifact=namedtuple("ModelTrainerArtifact",["trained_model_file_path","is_trained","message","train_rmse","test_rmse","train_accuracy","test_accuracy","model_accuracy"])
