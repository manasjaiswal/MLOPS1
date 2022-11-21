import os
import sys

from sales.exception.exception import SalesException
from sales.helper_functions.helper import loading_object
from sales.constant.constants import *
import pandas as pd

class SalesData:

    def __init__(self,
                Item_Fat_Content: str,
                Item_Identifier: str,
                Item_MRP: float,
                Item_Type: str,
                Item_Visibility: float,
                Item_Weight: float,
                Outlet_Establishment_Year: int,
                Outlet_Identifier: str,
                Outlet_Location_Type: str,
                Outlet_Size: str,
                Outlet_Type: str,
                Item_Outlet_Sales=None
                ):
        try:
            self.Item_Fat_Content=Item_Fat_Content
            self.Item_Identifier=Item_Identifier
            self.Item_MRP=Item_MRP
            self.Item_Outlet_Sales=Item_Outlet_Sales
            self.Item_Type=Item_Type
            self.Item_Visibility=Item_Visibility
            self.Item_Weight=Item_Weight
            self.Outlet_Establishment_Year=Outlet_Establishment_Year
            self.Outlet_Identifier=Outlet_Identifier
            self.Outlet_Location_Type=Outlet_Location_Type
            self.Outlet_Size=Outlet_Size
            self.Outlet_Type=Outlet_Type

        except Exception as e:
            raise SalesException()       

    def get_sales_input_data_frame(self)->pd.DataFrame:
        try:
            input_dict=self.get_data_as_dict()
            return pd.DataFrame(input_dict)
        except Exception as e:
            raise SalesException(e,sys) from e

    def get_data_as_dict(self)->dict:
        try:
            data={
                COLUMN_ITEM_IDENTIFIER:[self.Item_Identifier],
                COLUMN_ITEM_WEIGHT:[self.Item_Weight],
                COLUMN_FAT_CONTENT:[self.Item_Fat_Content],
                COLUMN_ITEM_VISIBILITY:[self.Item_Visibility],
                COLUMN_ITEM_TYPE:[self.Item_Type],
                COLUMN_ITEM_MRP:[self.Item_MRP],
                COLUMN_OUTLET_IDENTIFIER_KEY:[self.Outlet_Identifier],
                COLUMN_OUTLET_ESTABLISHMENT_YEAR:[self.Outlet_Establishment_Year],
                COLUMN_OUTLET_SIZE:[self.Outlet_Size],
                COLUMN_OUTLET_LOCATION_TYPE:[self.Outlet_Location_Type],
                COLUMN_OUTLET_TYPE:[self.Outlet_Type],
                }
            return data
        except Exception as e:
            raise SalesException(e,sys) from e        

class SalesPredictor:
    def __init__(self,model_dir:str):
        try:
            self.model_dir=model_dir
        except Exception as e:
            raise SalesException(e,sys) from e

    def get_latest_model_file_path(self)->str:
        try:
            li=os.listdir(self.model_dir)
            latest_model_dir=os.path.join(self.model_dir,li[-1])
            file_path=os.path.join(latest_model_dir,os.listdir(latest_model_dir)[0])
            return file_path
        except Exception as e:
            raise SalesException(e,sys) from e

    def predict(self,X):
        try:
            model_path=self.get_latest_model_file_path()
            model_obj=loading_object(model_path)
            return model_obj.predict(X)
        except Exception as e:
            raise SalesException(e,sys) from e
