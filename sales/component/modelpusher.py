from sales.exception.exception import SalesException
from sales.entity.config_entity import ModelPusherConfig
from sales.entity.artifact_entity import ModelEvaluationArtifact,ModelPusherArtifact
from sales.logger.logging import logging
import os,sys
import shutil

class ModelPusher:
    def __init__(self,model_evaluation_artifact:ModelEvaluationArtifact,model_pusher_config:ModelPusherConfig):
        try:
            logging.info(".............Model-pusher-log-started..........")
            self.model_evaluation_artifact=model_evaluation_artifact
            self.model_pusher_config=model_pusher_config
        except Exception as e:
            raise SalesException(e,sys) from e

    def get_model_pushing(self)->ModelPusherArtifact:
        try:
            model_evaluation_file_path=self.model_evaluation_artifact.evaluated_model_path
            model_export_dir=self.model_pusher_config.model_export_dir
            os.makedirs(model_export_dir,exist_ok=True)
            dst=os.path.join(model_export_dir,os.path.basename(model_evaluation_file_path))
            src=model_evaluation_file_path
            logging.info(f"Copying the evaluation_file_data from {src} to {dst}")
            shutil.copy(src=src,dst=dst)
            model_pusher_artifact=ModelPusherArtifact(
                is_model_pushed=True,
                export_model_file_path=dst
            )
            logging.info(f"Model Pusher Artifact:{model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise SalesException(e,sys) from e

    def initiate_model_pushing(self)->ModelPusherArtifact:
        try:
            return self.get_model_pushing()
        except Exception as e:
            raise SalesException(e,sys) from e        

    def __del__(self):
        logging.info("............Model-Pusher-Log-Completed............")        