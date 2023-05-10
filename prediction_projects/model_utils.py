import ast
import json
import pandas as pd
import pickle
import numpy as np
from google.cloud import storage
from google.auth import compute_engine
from datetime import datetime
import os
import sqlalchemy as sa


# LOAD DATA AND TRAINED MODEL #


def connect_redshift(credentials=None):
    # retrieve redshift credentials
    if credentials is None:
        # navigate to parent dir
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isdir(parent_dir):
            raise ValueError("Parent directory not found")

        # navigate to this dir
        # this_dir = os.path.dirname(os.path.abspath(__file__))

        # navigate to data dir
        dir_data = os.path.join(parent_dir, "data")
        if not os.path.isdir(dir_data):
            raise ValueError("Data directory not found")

        # navigate to secrets dir
        dir_secrets = os.path.join(dir_data, "secrets")
        if not os.path.isdir(dir_secrets):
            raise ValueError("Secrets directory not found")

        # navigate to file
        fn_connection = os.path.join(dir_secrets, "redshift_config.json")
        if not os.path.isfile(fn_connection):
            raise ValueError("Json file not found")

        # load file
        with open(fn_connection) as config_file:
            credentials = json.load(config_file)
        assert credentials is not None

    # connect Redshift
    engine = sa.create_engine(sa.engine.url.URL.create(drivername="postgresql+psycopg2",
                                                       username=credentials['user'],
                                                       password=credentials['password'],
                                                       host=credentials['host'],
                                                       port=credentials['port'],
                                                       database=credentials['dbname'],
                                                       )
                              ).connect()

    conn = engine.connection
    cur = conn.cursor()

    return engine, conn, cur


def split_channels(df):
    # calculate no of buckets to split
    batch_size = 1000
    n_k, n_m = divmod(len(df), batch_size)
    n_m = 1 if n_m > 0 else 0
    n = 1 if len(df) <= batch_size else n_k + n_m

    # calculate buckets size
    k, m = divmod(len(df), n)

    return (df[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def load_features_from_redshift(df, cur, config):
    print(f"Loading data from REDSHIFT - started at: \033[1m{datetime.now()}\033[0m")

    # retrieve pipeline_name from config file
    pipeline_name = config["Pipeline"].get("pipeline_name")

    # retrieve query from config file
    query = config["Redshift_Data"].get("query")

    # initialize empty df to store features data in it
    df_features = pd.DataFrame()

    # retrieve channels from input df (curation list)
    if pipeline_name == "Youtube_Search":
        channels_list = df['provider_id'].values.tolist()
    else:
        channels_list = df['boss_channel_id'].values.tolist()

    # add features for each channel
    for channels_bucket in split_channels(channels_list):
        if pipeline_name == "Youtube_Search":
            final_query = f"""{query}
                            and provider_id in ({",".join([f"'{channel}'" for channel in channels_bucket])})
                            """
        else:
            final_query = f"""{query}
                            and boss_channel_id in ({",".join([f"'{channel}'" for channel in channels_bucket])})
                            """
        cur.execute(final_query)
        df_query = pd.DataFrame(cur.fetchall())
        if df_query.empty:
            raise ValueError("The process 'load_features_from_redshift' failed because the query returned no results")
        df_query.columns = [desc[0] for desc in cur.description]
        df_features = pd.concat((df_features, df_query), axis=0)

    print(f"Loading data from REDSHIFT - finished at: \033[1m{datetime.now()}\033[0m")

    # join channels and features
    if pipeline_name == "Youtube_Search":
        df = df.merge(df_features, on='provider_id', how='left')
    else:
        df = df.merge(df_features, on='boss_channel_id')

    # drop duplications
    df = df.drop_duplicates()

    return df


def load_model_pkl_file(config):
    model_file_location = config["Locations"].get("model_file_location")
    file_name = config["GCS"].get("model_file_name")

    if model_file_location == "GCS" or model_file_location == "GCS_VIA_LOCAL":
        # configure paths
        gcs_bucket = config["GCS"].get("bucket")
        gcs_folder_path = config["GCS"].get("folder_path_of_model_file")
        gcs_file_path = str(gcs_folder_path) + str(file_name)
        # configure environment
        if model_file_location == "GCS":
            wi_credentials = compute_engine.Credentials()
            storage_client = storage.Client(credentials=wi_credentials)
        else:
            try:
                storage_client = storage.Client()
            except Exception as e:
                print("The error is: ", e)
                raise ValueError(
                    "Attempting to run the code in a local development environment - failed,"
                    " you need to run the following command (in CMD): gcloud auth application-default login")
        # access GCS bucket
        bucket = storage_client.bucket(gcs_bucket)
        # check if file exists
        is_file_exists = storage.Blob(bucket=bucket, name=gcs_file_path).exists(storage_client)
        if is_file_exists:
            blob = bucket.blob(gcs_file_path)
            # load the model (pkl file) from GCS
            blob.download_to_filename(file_name)
            with open(file_name, 'rb') as f:
                loaded_model = pickle.load(f)
            # extract from the trained model the columns (with order)
            cols_when_model_builds = loaded_model.feature_names_in_
        else:
            raise ValueError("Model file was not found in GCS bucket")

    elif model_file_location == "LOCAL":
        # load the model (pkl file) from local disk
        this_dir = os.path.dirname(os.path.abspath(__file__))
        my_path = f"{this_dir}/{file_name}"
        with open(my_path, 'rb') as f:
            loaded_model = pickle.load(f)
        # extract all the columns (with order) - from training dataset
        cols_when_model_builds = loaded_model.feature_names_in_

    else:
        raise ValueError("No location was specified in the config file")

    return loaded_model, cols_when_model_builds


# PROCESS DATA TO MATCH TRAINING DATASET #

def retrieve_model_bucketed_columns(cols_when_model_builds, config):
    # retrieve categorical features to bucket from config file
    features_to_bucket = ast.literal_eval(config["Data_Processing"].get("features_to_bucket"))

    # save buckets columns - from training dataset
    model_bucketed_columns = {}
    for feature in features_to_bucket:
        model_bucketed_columns[feature] = [col.replace(f"{feature}_bucket_", '') for col in cols_when_model_builds if
                                           f"{feature}" in col]

    return model_bucketed_columns


def bucket_df_rows(row, feature, model_bucketed_columns_values):
    if pd.isnull(row[feature]):
        return np.nan
    elif row[feature] in model_bucketed_columns_values:
        return row[feature]
    else:
        return 'Other'


def df_apply_bucket_transformation(df, model_bucketed_columns, config):
    # retrieve categorical features to bucket from config file
    features_to_bucket = ast.literal_eval(config["Data_Processing"].get("features_to_bucket"))

    # for every feature that should be bucketed - bucket it according to model buckets
    for feature in features_to_bucket:
        model_bucketed_columns_values = model_bucketed_columns[feature]
        df[feature] = df.apply(lambda row: bucket_df_rows(row, feature, model_bucketed_columns_values), axis=1)
        df = df.rename(columns={f"{feature}": f"{feature}_bucket"})

    return df


def prepare_data_for_trained_model(df,
                                   cols_when_model_builds,
                                   is_drop_cols,
                                   is_convert_numeric_cols_to_categorical,
                                   is_get_dummies,
                                   is_add_zeros_to_missing_cols,
                                   is_reorder_df_cols_to_match_trained_model,
                                   config
                                   ):
    if is_drop_cols:
        # retrieve col_to_drop from config file
        cols_to_drop = ast.literal_eval(config["Data_Processing"].get("cols_to_drop"))
        # drop columns
        df = df.drop(cols_to_drop, axis=1)

    if is_convert_numeric_cols_to_categorical:
        # retrieve numeric features to convert to categorical from config file
        numeric_to_category = ast.literal_eval(config["Data_Processing"].get("numeric_to_category"))
        # convert numeric features to categorical
        for feature in numeric_to_category:
            df[feature] = df[feature].astype(np.uint8)

    if is_get_dummies:
        # create dummy variables
        df = pd.get_dummies(df)

    if is_add_zeros_to_missing_cols:
        if len(df.columns) > len(cols_when_model_builds):
            print(
                f"\033[1mNote: The number of columns of the input "
                f"is greater than the number of columns of the trained model "
                f"---> Input will be missing {len(df.columns) - len(cols_when_model_builds)} columns.\033[0m")
        elif len(df.columns) < len(cols_when_model_builds):
            print(
                f"\033[1mNote: The number of columns of the input "
                f"is less than the number of columns of the trained model "
                f"---> Adding {len(cols_when_model_builds) - len(df.columns)} zero columns.\033[0m")
            # add missing columns (features) from training dataset - to the imported data (input df)
            missing_cols = [col for col in cols_when_model_builds if col not in list(df)]
            missing_cols_dict = dict.fromkeys(missing_cols, 0)
            missing_cols_df = pd.DataFrame(missing_cols_dict, index=df.index)
            df = pd.concat([df, missing_cols_df], axis=1)

    if is_reorder_df_cols_to_match_trained_model:
        # reorder the columns - to match the training dataset columns
        df = df[cols_when_model_builds].copy()

    return df


# CALIBRATE CLF PREDICTED PROBABILITIES #

def probability_calibration(df, config):
    # retrieve probability_calibration_type from config file
    calibration_model_type = config["ProbabilityCalibration"].get("calibration_model_type")

    if calibration_model_type:

        # load calibration model
        with open('probability_calibration_model.pkl', 'rb') as f:
            probability_calibration_model = pickle.load(f)

        # calibrate predictions
        if calibration_model_type == "IsotonicRegression":
            df['predicted_acceptance_rate'] = pd.DataFrame(
                probability_calibration_model.predict(df['predicted_acceptance_rate']))
        elif calibration_model_type == "CalibratedClassifierCV":
            df['predicted_acceptance_rate'] = pd.DataFrame(probability_calibration_model.predict_proba(
                df[df.columns[~df.columns.isin(['predicted_acceptance_rate'])]])).T[1]

        return df

    return df
