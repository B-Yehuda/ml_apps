import io
from io import StringIO
from datetime import datetime
import pandas as pd
from flask import Flask, make_response, request, render_template, Response

from ml_projects_utilities.load_config_utils import load_config

from prediction_projects.model_utils import \
    connect_redshift, load_features_from_redshift
from prediction_projects.predict import \
    create_tutorials_predictions_df, create_deposits_predictions_df, create_acceptance_rate_predictions_df

from youtube_lookalikes_project.youtube_search import community_based_search
from youtube_lookalikes_project.youtube_scraping_functions import build_youtube

from vault import get_vault_secrets

# INITIALIZE FLASK APP #

app = Flask(__name__)


# PRE-APPLICATION FUNCTIONS #

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


# CREATE WEB APPLICATIONS ROUTES #

# create home html page
@app.route('/', methods=['GET'])
def home():
    return render_template('index_home.html')


# create acceptance_rate html page
@app.route('/acceptance_rate', methods=['GET'])
def acceptance_rate_page():
    return render_template('index_acceptance_rate.html')


# create kronos html page
@app.route('/kronos', methods=['GET'])
def kronos_page():
    return render_template('index_kronos.html')


# create YouTube html page
@app.route('/youtube', methods=['GET'])
def youtube_search_page():
    return render_template('index_youtube_search.html')


# health check route
@app.route("/health")
def health_check():
    print("/health request")
    status_code = Response(status=200)
    return status_code


# CREATE WEB APPLICATIONS FUNCTIONS #

# create predict_acceptance_rate flask app
@app.route('/predict_acceptance_rate', methods=["POST"])
def predict_acceptance_rate():
    # request csv file
    f = request.files['data_file']
    if not f:
        return "No file"

    # read data from csv file
    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    stream.seek(0)
    result = transform(stream.read())

    # convert the data into df
    df_input = pd.read_csv(StringIO(result))

    # create df_acceptance_rate_predictions
    df_acceptance_rate_predictions, acceptance_rate_clf_config = \
        create_acceptance_rate_predictions_df(df_input=df_input,
                                              add_features_from_redshift_to_df_input=True
                                              )

    # return predictions
    response = make_response(df_acceptance_rate_predictions.to_csv(index=False))
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"

    return response


# create predict_kronos flask app
@app.route('/predict_kronos', methods=["POST"])
def predict_kronos():
    """
    Background:
        This is a prediction pipeline which predict 1st using the Tutorials trained model,
         then using the Deposits trained model.

    Usage:
        To switch between different pipelines, change the values of the "model_types" list in:
         create_acceptance_rate_predictions_df, create_tutorials_predictions_df functions, for example:
            * ["clf", "reg"] - predict using classifier and regressor hurdle models
            * ["reg"] / ["clf"] - predict using regressor / classifier single model respectively
    """

    # request csv file
    f = request.files['data_file']
    if not f:
        return "No file"

    # read data from csv file
    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    stream.seek(0)
    result = transform(stream.read())

    # convert the data into df
    df_input = pd.read_csv(StringIO(result))

    # create df_tutorials_predictions
    df_tutorials_predictions, is_previous_pipeline_clf_reg, tutorials_clf_config = \
        create_tutorials_predictions_df(df_input=df_input,
                                        add_features_from_redshift_to_df_input=True
                                        )

    # create df_tutorials_and_deposits_predictions
    df_tutorials_and_deposits_predictions = \
        create_deposits_predictions_df(df_input=df_tutorials_predictions,
                                       add_features_from_redshift_to_df_input=False,
                                       is_previous_pipeline_clf_reg=is_previous_pipeline_clf_reg
                                       )

    # return predictions
    response = make_response(df_tutorials_and_deposits_predictions.to_csv(index=False))
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"

    return response


# create youtube_search flask app
@app.route('/youtube_search', methods=["POST"])
def youtube_search():
    # request csv file
    f = request.files['data_file']
    if not f:
        return "No file"

    # read data from csv file
    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    stream.seek(0)
    result = transform(stream.read())

    # convert the data into df and drop duplications
    df_input = pd.read_csv(StringIO(result))
    df_input = df_input.drop_duplicates()

    # -------------------------- YOUTUBE PIPELINE -------------------------- #
    # define model names/types to use with it
    models_name = ["search"]
    model_types = ["youtube"]

    # retrieve config file
    config_objects = load_config(models_name, model_types)

    # load config
    for config in config_objects.values():
        youtube_config = config[0]

    # retrieve YouTube API
    if youtube_config["Locations"].get("model_file_location") == "GCS":
        credentials = get_vault_secrets()[0]
        api_key = get_vault_secrets()[1]["API_KEY"]
    elif youtube_config["Locations"].get("model_file_location") == "LOCAL":
        credentials = None
        api_key = None
    else:
        raise ValueError("No location was specified in the config file")

    # construct YouTube API object
    youtube = build_youtube(api_key=api_key)

    # iterate over channels url input and retrieve recommendations (relative channels)
    print(f"Youtube search process for {len(df_input)} channels - started at: \033[1m{datetime.now()}\033[0m")
    df_result = community_based_search(
        youtube=youtube,
        df_input=df_input,
        n_recommendations=int(youtube_config["Data_Processing"]["n_recommendations"]),
        n_videos_per_channel=int(youtube_config["Data_Processing"]["n_videos_per_channel"]),
        n_comments_per_video_of_channel=int(youtube_config["Data_Processing"]["n_comments_per_video_of_channel"]),
        n_videos_per_keyword=int(youtube_config["Data_Processing"]["n_videos_per_keyword"]),
        n_comments_per_video_of_keyword=int(youtube_config["Data_Processing"]["n_comments_per_video_of_keyword"]),
        keyword_target_length=int(youtube_config["Data_Processing"]["keyword_target_length"]),
        n_keywords=int(youtube_config["Data_Processing"]["n_keywords"])
    )
    print(f"Youtube search process for {len(df_input)} channels - finished at: \033[1m{datetime.now()}\033[0m")

    # prepare df to redshift (extract provider_id from given url --> so we can join bi_db.creators.provider_id)
    df_result["provider_id"] = df_result["Channel URL (Output)"].str.split('/').str[-1]

    # connect redshift
    cur = connect_redshift(credentials=credentials)[1]

    # add features to csv file
    df = load_features_from_redshift(df=df_result,
                                     cur=cur,
                                     config=youtube_config
                                     )

    # rename and reorder columns
    df.rename(columns={'boss_channel_id': 'BOSS Channel ID (Output)'}, inplace=True)
    df = df[["Channel URL (Input)", "Channel URL (Output)", "BOSS Channel ID (Output)", "Score"]]

    # return predictions
    response = make_response(df.to_csv(index=False))
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0")
