import logging
from datetime import datetime
import os
import pandas as pd
from sales.constant.constants import CURRENT_TIME_STAMP,LOG_DIR

LOG_FILE_NAME=f"log_{CURRENT_TIME_STAMP}.log"
LOG_FILE_PATH=os.path.join(LOG_DIR,LOG_FILE_NAME)
os.makedirs(LOG_DIR,exist_ok=True)

logging.basicConfig(filename=LOG_FILE_PATH,
filemode='w',
format='[%(asctime)s]^;%(levelname)s^;%(lineno)d^;%(filename)s^;%(funcName)s()^;%(message)s',
level=logging.INFO
)

def get_log_dataframe(file_path):
    data=[]
    with open(file_path) as log_file:
        for line in log_file.readlines():
            data.append(line.split("^;"))
    columns=["Time Stamp","Log Level","line number","file name","function name","message"]
    
    log_df=pd.DataFrame(data)

    log_df.columns=columns
    
    log_df["log_message"]=log_df['Time Stamp'].astype(str) +":$"+ log_df["message"]  

    return log_df[["log_message"]] 
