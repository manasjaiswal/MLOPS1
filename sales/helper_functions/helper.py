import os,sys
from sales.exception.exception import SalesException
import yaml
import dill
import pandas as pd
import numpy as np
from sales.constant.constants import *

def read_yaml_file(file_path)->dict:
    """
    This function reads yaml file
    """
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise SalesException(e,sys) from e 


def write_yaml_file(file_path,json_content:dict=None):
    """
    This function writes into yaml file
    """
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    try:
        with open(file_path,'w') as yaml_file:
            yaml.dump(json_content,yaml_file)

    except Exception as e:
        raise SalesException(e,sys) from e        

def load_data(file_path:str,schema_file_path:str):
    try:
        data_set_schema=read_yaml_file(schema_file_path)

        schema_data_types=data_set_schema[SCHEMA_CONFIG_DATA_TYPES_KEY]

        error_message=""
        
        data_frame=pd.read_csv(file_path)

        for column in data_frame.columns:
            if column in list(schema_data_types.keys()):
                data_frame[column].astype(schema_data_types[column])
            else:
                error_message=f"column {column} in dataframe not found in schema"
        if len(error_message)>0:
            raise Exception(e)
        else:
            return data_frame                
    except Exception as e:
        raise SalesException(e,sys)

def save_numpy_array_data(file_path:str,array:np.array):
    """"
    Saving numpy array data into .npz file
    """
    try:
        array=array.astype('float64')
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as numpy_file:
            np.save(numpy_file,array)
    except Exception as e:
        raise SalesException(e,sys)

def load_numpy_array_data(file_path)->np.ndarray:
    """
    Loading numpy array data
    """        
    try:
        with open(file_path,'rb') as numpy_file:
            return np.load(numpy_file,allow_pickle=True)
    except Exception as e:
        raise SalesException(e,sys) from e

def saving_object(file_path:str,obj):
    """
    Saving a .pkl object in file
    """
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise SalesException(e,sys) from e

def loading_object(file_path:str):
    """
    Loading a .pkl object from the file
    """
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise SalesException(e,sys) from e