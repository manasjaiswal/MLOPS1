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
import sys,os
from threading import Thread
from sales.constant.constants import EXPERIMENT_DIR_NAME,EXPERIMENT_FILE_NAME
import uuid
from datetime import datetime
from collections import namedtuple
import pandas as pd
import numpy as np

Experiment=namedtuple("Experiment",["experiment_id","initialization_time_stamp","artifact_time_stamp","running_status","start_time","stop_time",
                                    "execution_time","message","experiment_file_path","accuracy","is_model_accepted"])


class SalesPipeline(Thread):
    experiment:Experiment=Experiment(*([None]*11))
    experiment_file_path=None

    def __init__(self,config:Configuration):
        try:
            os.makedirs(config.training_pipeline_config.artifact_dir,exist_ok=True)
            SalesPipeline.experiment_file_path=os.path.join(config.training_pipeline_config.artifact_dir,EXPERIMENT_DIR_NAME,EXPERIMENT_FILE_NAME)
            super().__init__(daemon=False,name="sales_pieline")
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
            if SalesPipeline.experiment.running_status:
                logging.info("Pipeline is already runnig")
                return SalesPipeline.experiment
            logging.info("Pipeline starting")

            experiment_id=str(uuid.uuid4())

            SalesPipeline.experiment=Experiment(experiment_id=experiment_id, 
                                            initialization_time_stamp=self.config.time_stamp, 
                                            artifact_time_stamp=self.config.time_stamp, 
                                            running_status=True, 
                                            start_time=datetime.now(), 
                                            stop_time=None,
                                            execution_time=None, 
                                            message="Pipeline has been started", 
                                            experiment_file_path=SalesPipeline.experiment_file_path, 
                                            accuracy=None, 
                                            is_model_accepted=None) 

            logging.info(f"Pipeline experiment:{SalesPipeline.experiment}")    

            self.save_experiment()                             
            data_ingestion_artifact=self.data_ingestion_pipeline()
            data_validation_artifact=self.data_validation_pipeline(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.data_transformation_pipeline(data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact=data_validation_artifact)
            model_trainer_artifact=self.model_trainer_pipeline(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact=self.model_evaluation_pipeline(data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact=data_validation_artifact,model_trainer_artifact=model_trainer_artifact)
            
            if model_evaluation_artifact.is_model_accepted:
                model_pusher_artifact=self.model_pusher_pipeline(model_evaluation_artifact=model_evaluation_artifact)
                logging.info(f"Model Pusher Artifact:{model_pusher_artifact}")
            else:
                logging.info("Trianed Model Rejected")

            logging.info("Pipeline completed")    

            stop_time=datetime.now()
            SalesPipeline.experiment=Experiment(experiment_id=SalesPipeline.experiment.experiment_id, 
                                                initialization_time_stamp=SalesPipeline.experiment.initialization_time_stamp, 
                                                artifact_time_stamp=SalesPipeline.experiment.artifact_time_stamp, 
                                                running_status=False, 
                                                start_time=SalesPipeline.experiment.start_time, 
                                                stop_time=stop_time, 
                                                execution_time=stop_time-SalesPipeline.experiment.start_time, 
                                                message="Pipeline Has been completed", 
                                                experiment_file_path=SalesPipeline.experiment.experiment_file_path, 
                                                accuracy=model_trainer_artifact.model_accuracy, 
                                                is_model_accepted=model_evaluation_artifact.is_model_accepted
                                                )
            logging.info(f"SalesPipeline experiment:{SalesPipeline.experiment}") 
            self.save_experiment()
        except Exception as e:
            raise SalesException (e,sys) from e     


    def save_experiment(self):
        try:
            if SalesPipeline.experiment.experiment_id is not None:
                experiment=SalesPipeline.experiment
                experiment_dict=experiment._asdict()
                experiment_dict={key:[value] for key,value in experiment_dict.items()}
                experiment_dict.update({
                    "created_time_stamp":[datetime.now()],
                    "experiment_file_path":[os.path.basename(SalesPipeline.experiment.experiment_file_path)]
                })

                experiment_report=pd.DataFrame(experiment_dict)

                os.makedirs(os.path.dirname(SalesPipeline.experiment_file_path),exist_ok=True)
                if os.path.exists(SalesPipeline.experiment_file_path):
                    experiment_report.to_csv(SalesPipeline.experiment_file_path,index=False,header=False,mode="a")
                else:
                    experiment_report.to_csv(SalesPipeline.experiment_file_path,index=False,header=True,mode='w')

            else:
                print("Fisrst Start Experiment")
        except Exception as e:
            raise SalesException(e,sys) from e        

    def run(self):
        try:
            self.initiating_sales_pipeline()
        except Exception as e:
            raise SalesException(e,sys) from e

    @classmethod
    def get_experiments_status(cls,limit:int=5)->pd.DataFrame:
        try:
            if os.path.exists(SalesPipeline.experiment_file_path):
                df=pd.read_csv(SalesPipeline.experiment_file_path)
                limit=-1*int(limit)
                return df[limit:].drop(columns=["experiment_file_path","initialization_time_stamp"],axis=1)
            else:
                return pd.DataFrame    
        except Exception as e:
            raise SalesException(e,sys) from e        