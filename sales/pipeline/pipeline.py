from sales.exception.exception import SalesException
from sales.config.configuration import Configuration
from sales.logger.logging import logging
from sales.component.dataingestion import DataIngestion
from sales.component.datavalidation import DataValidation
from sales.component.datatranformation import DataTransformation
from sales.component.modeltrainer import ModelTrainer
from sales.component.modelevaluation import ModelEvaluation
from sales.component.modelpusher import ModelPusher
from sales.entity.config_entity import *
from sales.entity.artifact_entity import *
import sys

class SalesPipeline:
    def __init__(self,config:Configuration):
        try:
            self.config=config
        except Exception as e:
            raise SalesException(e,sys) from e

    def data_ingestion_pipeline(self)->DataIngestionArtifact:
        try:
            data_ingestion=DataIngestion(dataingestionconfig=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise SalesException (e,sys) from e

    def data_validation_pipeline(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=self.config.get_data_validation_config())
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise SalesException (e,sys) from e

    def data_transformation_pipeline(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        try:
            data_transformation=DataTransformation(data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact=data_validation_artifact,data_transformation_config=self.config.get_data_tranformation_config())
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise SalesException (e,sys) from e

    def model_trainer_pipeline(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            model_trainer=ModelTrainer(data_transformation_artifact=data_transformation_artifact,model_trainer_confg=self.config.get_model_trainer_config())
            return model_trainer.initiate_model_training()
        except Exception as e:
            raise SalesException (e,sys) from e

    def model_evaluation_pipeline(self,data_ingestion_artifact: DataIngestionArtifact, model_trainer_artifact: ModelTrainerArtifact, data_validation_artifact: DataValidationArtifact)->ModelEvaluationArtifact:
        try:
            model_evaluation=ModelEvaluation(data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact=data_validation_artifact,model_trainer_artifact=model_trainer_artifact,model_evaluation_config=self.config.get_model_evaluation_config())
            return model_evaluation.initiate_model_evaluation()
        except Exception as e:
            raise SalesException (e,sys) from e                               

    def model_pusher_pipeline(self,model_evaluation_artifact:ModelEvaluationArtifact):
        try:
            model_pusher=ModelPusher(model_evaluation_artifact=model_evaluation_artifact,model_pusher_config=self.config.get_model_pusher_config())
            model_pusher.initiate_model_pushing()
        except Exception as e:
            raise SalesException (e,sys) from e         

    def initiating_sales_pipeline(self):
        try:
            data_ingestion_artifact=self.data_ingestion_pipeline()
            data_validation_artifact=self.data_validation_pipeline(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.data_transformation_pipeline(data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact=data_validation_artifact)
            model_trainer_artifact=self.model_trainer_pipeline(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact=self.model_evaluation_pipeline(data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact=data_validation_artifact,model_trainer_artifact=model_trainer_artifact)
            self.model_pusher_pipeline(model_evaluation_artifact=model_evaluation_artifact)

        except Exception as e:
            raise SalesException (e,sys) from e     