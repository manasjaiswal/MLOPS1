from sales.exception.exception import SalesException
from sales.logger.logging import logging
import pandas as pd
import numpy as np
import os,sys
from sales.entity.config_entity import DataTransformationConfig
from sales.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact
from sales.helper_functions.helper import *
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sales.constant.constants import *
from sklearn.compose import ColumnTransformer
from typing import List
"""
This class Feature Handling will handle the 
1.ItemVisibilityValues which are zero because that
cannot be zero ,so we will change those to that of average of visibity according to item type
2.ItemWeight null values 
3.Outlet_Size null values
4.Item_Fat_Content replacement(Low Fat=LF,Regular=reg,low fat=LF)
"""
class FeatureHandling(BaseEstimator,TransformerMixin):
    def __init__(self,columns=None):
        try:
            self.columns=columns
        except Exception as e:
            raise SalesException(e,sys) from e

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        try:
            df=pd.DataFrame(X,columns=self.columns)
            #filling nan values of item_Weight with mean of the item_weight_values of the corresponding item type categories
            df[COLUMN_ITEM_WEIGHT]=df[COLUMN_ITEM_WEIGHT].fillna(df.groupby(COLUMN_ITEM_TYPE)[COLUMN_ITEM_WEIGHT].transform("mean"))
            #Filling nan values of the Outlet Size column with the most frequent
            df[COLUMN_OUTLET_SIZE]=df[COLUMN_OUTLET_SIZE].fillna(df[COLUMN_OUTLET_SIZE].mode()[0])
            #Replacement for Item Fat Content Categories
            df[COLUMN_FAT_CONTENT]=df[COLUMN_FAT_CONTENT].replace({'Low Fat':'LF','Regular':'reg','low fat':'LF'})
            di=df.groupby(COLUMN_ITEM_TYPE)[COLUMN_ITEM_VISIBILITY].median().to_dict()
            si=list(df[df[COLUMN_ITEM_VISIBILITY]==0].index)
            for i in si:
                df[COLUMN_ITEM_VISIBILITY].loc[i]=di[df[COLUMN_ITEM_TYPE].loc[i]]
            return df    
        except Exception as e:
            raise SalesException(e,sys) from e 

"""                    
This is for encoding for encoding categorical columns on the basis on the median of Item Outlet_Sales(Target_Variable) for each category
"""
class RankEncoder(BaseEstimator,TransformerMixin):
    def __init__(self,cat_columns,columns=None):
        self.columns=columns
        self.cat_columns=cat_columns
    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        try:
            cols=self.columns
            data=pd.DataFrame(X,columns=cols)
            cat_columns=self.cat_columns
            data[COLUMN_ITEM_OUTLET_SALES]=y
            data[COLUMN_ITEM_OUTLET_SALES]=data[COLUMN_ITEM_OUTLET_SALES].astype(float)
            for feature in cat_columns:
                di=data.groupby(feature)[COLUMN_ITEM_OUTLET_SALES].median().to_dict()
                e={}
                a=1
                for i in sorted(di.items(),key=lambda x:x[1],reverse=True):
                    e[i[0]]=a
                    a+=1    
                data[feature]=data[feature].map(e)
            return data.drop(columns=[COLUMN_ITEM_OUTLET_SALES])   
        except Exception as e:
            raise SalesException(e,sys) from e

"""
This is for the generation of two features Outlet_Year=2015-Outlet_Establishment_Year
and also Item_Identifier has only 2 indexes important for ex FDA15 we will consider only FD(Foods)
"""
class FeatureGenerator(BaseEstimator,TransformerMixin):
    def __init__(self,columns=None):
        self.columns=columns

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        try:
            data=pd.DataFrame(X,columns=self.columns)
            data[COLUMN_OUTLET_ESTABLISHMENT_YEAR]=data[COLUMN_OUTLET_ESTABLISHMENT_YEAR].apply(lambda x: 2015-x)
            data[COLUMN_ITEM_IDENTIFIER]=data[COLUMN_ITEM_IDENTIFIER].apply(lambda x: str(x)[0:2])
            return data  
        except Exception as e:
            raise SalesException(e,sys) from e


class DataTransformation:

    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
        try:
            logging.info("................Data Transformation Log Started..........")
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config
        except Exception as e:
            raise SalesException(e,sys) from e

    def get_data_transformation_object(self)->Pipeline:
        try:
            schema_info=read_yaml_file(file_path=self.data_validation_artifact.schema_file_path)
            train_df=load_data(file_path=self.data_ingestion_artifact.train_file_path,schema_file_path=self.data_validation_artifact.schema_file_path)
            numerical_columns=schema_info[SCHEMA_CONFIG_NUMERICAL_COLUMNS_KEY]
            categorical_columns=schema_info[SCHEMA_CONFIG_CATEGORICAL_COLUMNS_KEY]
            a=list(train_df.columns)
            a.remove(COLUMN_ITEM_OUTLET_SALES)
            tr1=ColumnTransformer([('feature_handling',FeatureHandling(columns=a),list(range(len(a))))])
            tr2=ColumnTransformer([('feature_generator',FeatureGenerator(columns=a),list(range(len(a))))])
            tr3=ColumnTransformer([('RankEncoder',RankEncoder(columns=a,cat_columns=categorical_columns),list(range(len(a))))])
            tr4=ColumnTransformer([('standardscaling',StandardScaler(with_mean=False),list(range(len(a))))])

            preprocessing=Pipeline(steps=[
                ('tr1',tr1),
                ('tr2',tr2),
                ('tr3',tr3),
                ('tr4',tr4)]
            )
            return preprocessing
        except Exception as e:
            raise SalesException(e,sys) from e

    def initiate_data_transformation(self)-> DataTransformationArtifact:
        try:
            logging.info("Getting the preprocessing object")
            preprocessing_obj=self.get_data_transformation_object()

            logging.info("Obtaining training and testing file paths")
            train_file_path=self.data_ingestion_artifact.train_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path
            schema_file_path=self.data_validation_artifact.schema_file_path
            
            logging.info("Loading test and train data frames")
            train_df=load_data(file_path=train_file_path,schema_file_path=schema_file_path)
            test_df=load_data(file_path=test_file_path,schema_file_path=schema_file_path)
            schema_info=read_yaml_file(file_path=schema_file_path)
            target_column=schema_info[SCHEMA_CONFIG_TARGET_COLUMN_KEY]
            
            logging.info("Applying preprocessing object on the training and testing data frame INPUT FEATURES")
            train_df_preprocessed_INPUT=preprocessing_obj.fit_transform(train_df.drop(columns=[target_column]),y=train_df[target_column])
            test_df_preprocessed_INPUT=preprocessing_obj.transform(test_df.drop(columns=[target_column]))
            train_df_preprocessed=np.c_[train_df_preprocessed_INPUT,np.array(train_df[target_column])]
            test_df_preprocessed=np.c_[test_df_preprocessed_INPUT,np.array(test_df[target_column])]

            logging.info("Saving test and train transformed arrays in transformed file paths")
            transformed_train_dir=self.data_transformation_config.transformed_train_dir
            transformed_test_dir=self.data_transformation_config.transformed_test_dir

            train_file_name=os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name=os.path.basename(test_file_path).replace(".csv",".npz")

            os.makedirs(transformed_train_dir,exist_ok=True)
            os.makedirs(transformed_test_dir,exist_ok=True)

            transformed_train_file_path=os.path.join(transformed_train_dir,train_file_name)
            transformed_test_file_path=os.path.join(transformed_test_dir,test_file_name)

            save_numpy_array_data(file_path=transformed_train_file_path,array=train_df_preprocessed)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_df_preprocessed)

            logging.info("Saving preprocessing object in the file for further use")
            preprocessing_dir=self.data_transformation_config.preprocessing_dir
            os.makedirs(preprocessing_dir,exist_ok=True)
            preprocessing_obj_file_path=os.path.join(preprocessing_dir,self.data_transformation_config.preprocessing_object_file_name)

            saving_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact=DataTransformationArtifact(
                transformed_train_dir_path=transformed_train_file_path, 
                transformed_test_dir_path=transformed_test_file_path, 
                preprocessed_object_file_path=preprocessing_obj_file_path,
                message="data_transformation_succesful"
            )
            logging.info(f"Data Transformation Artifact:{data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise SalesException(e,sys) from e                                

    def __del__(self):
        logging.info(".................Data-TransformationLog-Ended.........")

