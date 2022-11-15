from sales.exception.exception import SalesException
from sales.logger.logging import logging
import pandas as pd
import numpy as np
import os,sys
from sales.entity.config_entity import DataTransformationConfig
from sales.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact



class DataTransformation:

    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
        try:
            logging.info("................Data Transformation Log Started..........")
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config
        except Exception as e:
            raise SalesException(e,sys) from e

    def A(self):
        try:
            pass
        except Exception as e:
            raise SalesException(e,sys) from e

    def A(self):
        try:
            pass
        except Exception as e:
            raise SalesException(e,sys) from e

    def A(self):
        try:
            pass
        except Exception as e:
            raise SalesException(e,sys) from e

    def A(self):
        try:
            pass
        except Exception as e:
            raise SalesException(e,sys) from e

    def initiate_data_transformation(self)-> DataTransformationArtifact:
        try:
            pass
        except Exception as e:
            raise SalesException(e,sys) from e                                

    def __del__(self):
        logging.info(".................Data-TransformationLog-Ended.........")

