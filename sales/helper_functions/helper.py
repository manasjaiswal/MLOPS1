import os,sys
from sales.exception.exception import SalesException
import yaml
import dill
import pandas as pd
import numpy as np
from sales.constant.constants import *

def read_yaml_file(file_path):
    """
    This function reads yaml file
    """
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise SalesException(e,sys) from e 


def write_yaml_file(file_path,json_content:dict):
    """
    This function writes into yaml file
    """
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    try:
        with open(file_path,'w') as yaml_file:
            yaml.dump(json_content,yaml_file)

    except Exception as e:
        raise SalesException(e,sys) from e        