from sales.config.configuration import Configuration
from sales.pipeline.pipeline import SalesPipeline
from sales.exception.exception import SalesException
import sys

class Demo:
    def __init__(self):
        try:
            self.config=Configuration()
        except Exception as e:
            raise SalesException(e,sys) from e

    def demo_run_pipeline(self):
        try:
            pipeline=SalesPipeline(config=self.config)
            pipeline.run()
        except Exception as e:
            raise SalesException(e,sys) from e    


demo=Demo()
demo.demo_run_pipeline()            