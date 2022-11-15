from flask import Flask
from sales.logger.logging import logging
from sales.exception.exception import SalesException
import sys
app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    try:
        logging.INFO('hi there')
        return 'CI/CD structure done'
    except Exception as e:
        SalesException(e,sys)

if __name__=='__main__':
    app.run(debug=True)



  