from flask import Flask,request
from sales.logger.logging import logging,get_log_dataframe
from sales.exception.exception import SalesException
import sys,os
import pip
from sales.helper_functions.helper import read_yaml_file,write_yaml_file
from matplotlib.style import context
import json
from sales.config.configuration import Configuration
from sales.constant.constants import CONFIG_DIR,get_current_time_stamp
from sales.pipeline.pipeline import SalesPipeline
from flask import send_file,abort,render_template
from sales.entity.sales_predictor import SalesData,SalesPredictor


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "sales_logs"
PIPELINE_FOLDER_NAME = "sales"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


SALES_DATA_KEY = "sales_data"
ITEM_SALES_VALUE_KEY = "Item_Outlet_Sales"

app = Flask(__name__)


@app.route('/artifact', defaults={'req_path': 'sales'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("sales", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = SalesPipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = SalesPipeline(config=Configuration(current_time_stamp=get_current_time_stamp()))
    if not SalesPipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": SalesPipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        SALES_DATA_KEY: None,
        ITEM_SALES_VALUE_KEY: None
    }

    if request.method == 'POST':
        item_identifier = (request.form['item_identifier'])
        item_type = (request.form['item_type'])
        item_weight = float(request.form['item_weight'])
        outlet_type = (request.form['outlet_type'])
        outlet_location_type = (request.form['outlet_location_type'])
        item_visibility = float(request.form['item_visibility'])
        outlet_establishment_year = float(request.form['outlet_establishment_year'])
        outlet_identifier = (request.form['outlet_identifier'])
        item_mrp = float(request.form['item_mrp'])
        outlet_size=(request.form['outlet_size'])
        fat_content=(request.form['fat_content'])

        sales_data = SalesData(Item_Fat_Content=fat_content,
                                Item_Identifier=item_identifier, 
                                Item_MRP=item_mrp, 
                                Item_Type=item_type, 
                                Item_Visibility=item_visibility, 
                                Item_Weight=item_weight, 
                                Outlet_Establishment_Year=outlet_establishment_year, 
                                Outlet_Location_Type=outlet_location_type, 
                                Outlet_Size=outlet_size, 
                                Outlet_Type=outlet_type,
                                Outlet_Identifier=outlet_identifier, 
                                )
        sales_df = sales_data.get_sales_input_data_frame()
        sales_predictor = SalesPredictor(model_dir=MODEL_DIR)
        item_outlet_sales = sales_predictor.predict(X=sales_df)
        context = {
            SALES_DATA_KEY: sales_data.get_data_as_dict(),
            ITEM_SALES_VALUE_KEY: item_outlet_sales,
        }
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except  Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
    app.run()
  