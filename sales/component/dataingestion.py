from sales.exception.exception import SalesException
from sales.logger.logging import logging
import os,sys
from sales.config.configuration import Configuration
from sales.entity.config_entity import DataIngestionConfig
from sales.entity.artifact_entity import DataIngestionArtifact
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

class DataIngestion:

    def __init__(self,dataingestionconfig:DataIngestionConfig):
        try:
            logging.info(".............DataIngestionLogStarted.............")
            self.dataingestionconfig=dataingestionconfig
        except Exception as e:
            raise SalesException(e,sys) from e

    def splitting_training_dataset(self) -> DataIngestionArtifact:
        try:
            train_file_path=os.path.join(self.dataingestionconfig.data_dir,
            self.dataingestionconfig.train_file_name
            )
            os.makedirs(self.dataingestionconfig.data_dir,exist_ok=True)

            logging.info(f"Reading dataset in csv file:{self.dataingestionconfig.train_file_name}")
            if os.path.exists(train_file_path):
                sales_df=pd.read_csv(train_file_path)
            else:
                raise SalesException("Dataset_Missing",sys)    
            prediction_file_path=os.path.join(self.dataingestionconfig.data_dir,self.dataingestionconfig.test_file_name)
            """
            We will divide the train dataset into training and testing 
            because we dont have outputs for the testing dataset
            we will use the given test dataset for our predictions
            """
            sales_df["MRP_category"]=pd.cut(
                sales_df["Item_MRP"],
                bins=[0,50,100,150,200,250,np.inf],
                labels=[1,2,3,4,5,6]
            )

            logging.info(f"Splitting data into test and train")
            strat_train_set=None
            strat_test_set=None

            split= StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

            for train_index,test_index in split.split(sales_df,sales_df["MRP_category"]):
                strat_train_set=sales_df.loc[train_index].drop(["MRP_category"],axis=1)
                strat_test_set=sales_df.loc[test_index].drop(["MRP_category"],axis=1)

            ingested_train_file_path=os.path.join(self.dataingestionconfig.ingested_train_dir,self.dataingestionconfig.train_file_name)
            ingested_test_file_path=os.path.join(self.dataingestionconfig.ingested_train_dir,self.dataingestionconfig.test_file_name)

            if strat_train_set is not None:
                os.makedirs(self.dataingestionconfig.ingested_train_dir,exist_ok=True)
                logging.info(f"Copying splitted train data into file:{ingested_train_file_path}")
                strat_train_set.to_csv(ingested_train_file_path,index=False)

            if strat_test_set is not None:
                os.makedirs(self.dataingestionconfig.ingested_test_dir,exist_ok=True)
                logging.info(f"Copying splitted train data into file:{ingested_test_file_path}")
                strat_train_set.to_csv(ingested_test_file_path,index=False)

            data_ingestion_artifact=DataIngestionArtifact(
                train_file_path=ingested_train_file_path, 
                test_file_path=ingested_test_file_path, 
                prediction_file_path=prediction_file_path,
                is_ingested=True, 
                message="data_ingestion_done"
            )
            logging.info(f"Data IngestionArtifact {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise SalesException(e,sys) from e

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            return self.splitting_training_dataset()
        except Exception as e:
            raise SalesException(e,sys) from e

    def __del__(self):
        logging.info("...............DataIngestion-Log-Ended...................")