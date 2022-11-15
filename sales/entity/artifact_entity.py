from collections import namedtuple


#DataIngestionArtifact

DataIngestionArtifact=namedtuple("DataIngestionArtifact",["train_file_path","test_file_path","prediction_file_path","is_ingested","message"])

#DataValidationArtifact

DataValidationArtifact=namedtuple("DataValidationArtifact",["report_file_path","report_page_file_path","is_validated","null_existence","outlier_existence","data_drift_existence"])

#DataTransformationArtifact

DataTransformationArtifact=namedtuple("DataTransformationArtifact",["transformed_train_dir_path","transformed_test_dir_path","preprocessed_object_file_path"])