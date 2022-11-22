from cmath import log
import importlib
from pyexpat import model
import numpy as np
from sales.exception.exception import SalesException
import os,sys

from collections import namedtuple
from typing import List
from sales.logger.logging import logging
from sklearn.metrics import r2_score,mean_squared_error,confusion_matrix,fbeta_score,roc_auc_score
from sales.constant.constants import *
from sales.helper_functions.helper import *

InitializedModelDetail=namedtuple("InitializedModelDetail",
                                ["model_serial_number","model","param_grid_search","model_name"])

GridSearchedBestModel=namedtuple("GridSearchedBestModel",
                                ["model_serial_number","model","best_model","best_parameters","best_score"])

BestModel=namedtuple("BestModel",
                    ["model_serial_number","model","best_model","best_parameters","best_score"])

MetricInfoArtifact=namedtuple("MetricInfoArtifact",
                            ["model_name","model_object","train_rmse","test_rmse","train_accuracy","test_accuracy","model_accuracy","index_number"])

MetricInfoArtifact2=namedtuple("MetricInfoArtifact2",
                            ["model_name","model_object","train_fbeta","test_fbeta","model_fbeta","train_accuracy","test_accuracy","model_accuracy","index_number"])

def get_sample_model_config_yaml_file(export_dir:str)->str:
    try:
        model_config={
            GRID_SEARCH_KEY:{
                MODULE_KEY:"sklearn.model_selection",
                CLASS_KEY:"GridSearchCV",
                PARAM_KEY:{
                    "cv":3,
                    "verbose":1
                }
            },
            MODEL_SELECTION_KEY:{
                "module_0":{
                    MODULE_KEY:"module_of_model",
                    CLASS_KEY:"class_of_model",
                    PARAM_KEY:{
                        "params1":"value1",
                        "params2":"value2"
                    },
                    SEARCH_PARA_GRID_KEY:{
                        "param_name":['param_value_1','param_value_2']
                    }
                },
                "module_1":{
                    MODULE_KEY:"module_of_model",
                    CLASS_KEY:"class_of_model",
                    PARAM_KEY:{
                        "params1":"value1",
                        "params2":"value2"
                    },
                    SEARCH_PARA_GRID_KEY:{
                        "param_name":['param_value_1','param_value_2']
                    }
                }
            }
        }
        os.makedirs(export_dir,exist_ok=True)
        path=os.path.join(export_dir,"model.yaml")
        write_yaml_file(file_path=path,json_content=model_config)
        return path
    except Exception as e:
        raise SalesException(e,sys) from e

def evaluate_classification_model(model_list:list,X_train:np.ndarray,y_train:np.ndarray,X_test:np.ndarray,y_test:np.ndarray,base_accuracy:float=0.5)->MetricInfoArtifact2:
    """
    This function compares multiple classification models and returns the best model
    """
    try:
        metric_info_artifact=MetricInfoArtifact(
            model_name, 
            model_object, 
            train_fbeta, 
            test_fbeta,

            train_accuracy, 
            test_accuracy, 
            model_accuracy, 
            index_number
        )
        return metric_info_artifact
    except Exception as e:
        raise SalesException(e,sys) from e

def evaluate_regression_model(model_list:list,X_train:np.ndarray,y_train:np.ndarray,X_test:np.ndarray,y_test:np.ndarray,base_accuracy:float=0.5)->MetricInfoArtifact:
    """
    Description:
    This function compares multiple regression model returns best model
    """
    try:
        index_no=0
        metric_info_artifact=None
        for model in model_list:
            model_name=str(model) #getting model name based on object
            logging.info(f"..........Starting evaluating Model:[{type(model).__name__}]...................")

            #Getting prediction on training and testing dataset
            y_train_pred=(model.predict(X_train))
            y_test_pred=(model.predict(X_test))
    
            #Getting r squared error on training and testing dataset
            train_acc=r2_score(y_train,y_train_pred)
            test_acc=r2_score(y_test,y_test_pred)

            #getting train and test rmse
            train_rmse=np.sqrt(mean_squared_error(y_train,y_train_pred))
            test_rmse=np.sqrt(mean_squared_error(y_test,y_test_pred))

            #Calculating model accuracy as harmonic mean of test and train
            model_accuracy=(2*train_acc*test_acc)/(train_acc+test_acc)
            diff_test_train_accuracy=abs(test_acc-train_acc)

            #logging the import details of currently evaluated model
            logging.info(f"Score:{model_accuracy}")
            logging.info(f"Train_rmse:{train_rmse}")
            logging.info(f"Test_rmse:{test_rmse}")
            logging.info(f"Train_acc:{train_acc}")
            logging.info(f"Test_acc:{test_acc}")

            if model_accuracy>=base_accuracy and diff_test_train_accuracy<TOLERANCE_LIMIT:
                base_accuracy=model_accuracy
                metric_info_artifact=MetricInfoArtifact(
                    model_name=model_name, 
                    model_object=model, 
                    train_rmse=train_rmse, 
                    test_rmse=test_rmse, 
                    train_accuracy=train_acc, 
                    test_accuracy=test_acc, 
                    model_accuracy=model_accuracy, 
                    index_number=index_no
                )
            index_no+=1
        if metric_info_artifact is None:
            logging.info(f"No model found with higher accuracy than base model")    
        else:
            logging.info(f"Acceptable model found: {metric_info_artifact}")

        return metric_info_artifact       
    except Exception as e:
        raise SalesException(e,sys) from e
class ModelFactory:
    def __init__(self,model_config_path:str=None):
        try:
            self.config:dict=ModelFactory.read_params(model_config_path)
            self.grid_search_cv_module:str=self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name:str=self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_param_mapper:dict=dict(self.config[GRID_SEARCH_KEY][PARAM_KEY])
            self.models_initialization_config:dict=dict(self.config[MODEL_SELECTION_KEY])
            self.initialized_model_list=None
            self.grid_searched_best_model_list=None

        except Exception as e:
            raise SalesException(e,sys) from e

    @staticmethod
    def read_params(config_path:str)->dict:
        try:
            return read_yaml_file(file_path=config_path)
        except Exception as e:
            raise SalesException(e,sys) from e

    @staticmethod
    def class_for_name(module_name:str,class_name:str):
        try:
            module=importlib.import_module(module_name)
            logging.info(f"Executing command from {module} import {class_name}")
            class_ref=getattr(module,class_name)
            return class_ref
        except Exception as e:
            raise SalesException(e,sys) from e

    @staticmethod
    def update_property_of_class(instance_ref:object,mapper_data:dict)->object:
        try:
            if not isinstance(mapper_data,dict):
                raise Exception("Parameter details Required")
            for key,value in mapper_data.items():
                logging.info(f"Executing {str(instance_ref)}.{key}={value}")
                setattr(instance_ref,key,value)
            return instance_ref        
        except Exception as e:
            raise SalesException(e,sys) from e

    def execute_grid_search_operation(self,initialized_model: InitializedModelDetail,input_feature,output_feature)-> GridSearchedBestModel:
        """
        this will perform parameter search operation and will return the best optimistic model with best parameter
        estimator:Model object
        returns Grid SearchOperation object
        """
        try:
            # instanciating GridSearchCV class

            grid_search_cv_ref=ModelFactory.class_for_name(module_name=self.grid_search_cv_module,class_name=self.grid_search_class_name)
            grid_search_cv=grid_search_cv_ref(estimator=initialized_model.model,param_grid=initialized_model.param_grid_search)

            grid_search_cv=ModelFactory.update_property_of_class(instance_ref=grid_search_cv,mapper_data=self.grid_search_param_mapper)

            logging.info(f"Training {type(initialized_model.model).__name__} Started")
            grid_search_cv.fit(input_feature,output_feature)
            logging.info(f"Training {type(initialized_model.model).__name__} Completed")
            grid_searched_best_model=GridSearchedBestModel(
                model_serial_number=initialized_model.model_serial_number, 
                model=initialized_model.model, 
                best_model=grid_search_cv.best_estimator_, 
                best_parameters=grid_search_cv.best_params_, 
                best_score=grid_search_cv.best_score_
            )
            return grid_searched_best_model
        except Exception as e:
            raise SalesException(e,sys) from e 

    def get_initialized_model_list(self) -> List[InitializedModelDetail]:
        """
        This function returns the list of initialized model list
        """
        try:
            model_info=self.models_initialization_config
            initialized_model_list=[]
            for item in list(model_info.keys()):
                model_serial_no=item
                model_ref=ModelFactory.class_for_name(module_name=model_info[item][MODULE_KEY],class_name=model_info[item][CLASS_KEY])
                model=model_ref()
                if PARAM_KEY in model_info[item]:
                    model=ModelFactory.update_property_of_class(instance_ref=model,mapper_data=dict(model_info[item][PARAM_KEY]))    
                param_grid_search_mapper=model_info[item][SEARCH_PARA_GRID_KEY]
                model_name=f"{type(model).__name__}"
                initialized_model_detail=InitializedModelDetail(
                    model_serial_number=model_serial_no, 
                    model=model, 
                    param_grid_search=param_grid_search_mapper,
                    model_name=model_name
                )
                initialized_model_list.append(initialized_model_detail)
            self.initialized_model_list=initialized_model_list
            return self.initialized_model_list
        except Exception as e:
            raise SalesException(e,sys) from e

    def initiate_best_parameter_search_for_initialized_model(self,initialized_model_detial:InitializedModelDetail,input_feature,output_feature)->GridSearchedBestModel:
        """
        This function returns a grid searched best model
        """
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model_detial,input_feature=input_feature,output_feature=output_feature)
        except Exception as e:
            raise SalesException(e,sys) from e

    def initiate_best_parameter_search_for_initialized_models(self,initialized_model_list:List[InitializedModelDetail],input_feature,output_feature)->List[GridSearchedBestModel]:
        """
        This function returns the grid searched best model list
        """
        try:
            grid_searched_best_model_list=[]
            for initialized_model_detail in initialized_model_list:
                grid_searched_best_model=self.initiate_best_parameter_search_for_initialized_model(initialized_model_detial=initialized_model_detail,input_feature=input_feature,output_feature=output_feature)
                grid_searched_best_model_list.append(grid_searched_best_model)
            self.grid_searched_best_model_list=grid_searched_best_model_list    
            return self.grid_searched_best_model_list
        except Exception as e:
            raise SalesException(e,sys) from e        

    @staticmethod
    def get_model_detail(model_details:List[InitializedModelDetail],model_serial_number:str)-> InitializedModelDetail:
        """
        This function returns the initialized model details at the specified model serial number
        """        
        try:
            for model_data in model_details:
                if model_data.model_serial_number==model_serial_number:
                    return model_data
        except Exception as e:
            raise SalesException(e,sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list:List[GridSearchedBestModel],base_accuracy:float=0.5)->BestModel:
        """
        This function returns the best model from the gridsearched best model list
        """
        try:
            best_model=None
            for grid_searched_best_model_detail in grid_searched_best_model_list:
                if grid_searched_best_model_detail.best_score>=base_accuracy:
                    logging.info(f"Acceptable Model found :{grid_searched_best_model_detail}")
                    base_accuracy=grid_searched_best_model_detail.best_score
                    best_model=grid_searched_best_model_detail
            if not best_model:
                raise Exception("None of the models have higher accuracy than base accuracy")
            logging.info(f"Best Model:{best_model}")
            return best_model
        except Exception as e:
            raise SalesException(e,sys) from e

    def get_best_model(self,X,y,base_accuracy=0.5)->BestModel:
        try:
            initialized_model_list=self.get_initialized_model_list()  
            print(initialized_model_list)
            logging.info(f"Initialized model list:{initialized_model_list}")
            grid_searched_best_model_list=self.initiate_best_parameter_search_for_initialized_models(initialized_model_list=initialized_model_list,input_feature=X,output_feature=y)          
            logging.info(f"Grid Searched best model list:{grid_searched_best_model_list}")
            best_model=ModelFactory.get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list=grid_searched_best_model_list,base_accuracy=base_accuracy)  
            return best_model
        except Exception as e:
            raise SalesException(e,sys) from e
