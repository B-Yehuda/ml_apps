import ast
from datetime import datetime
import numpy as np
import pandas as pd

from ml_projects_utilities.load_config_utils import load_config

from prediction_projects.model_utils import \
    load_model_pkl_file, retrieve_model_bucketed_columns, \
    df_apply_bucket_transformation, prepare_data_for_trained_model, \
    probability_calibration, connect_redshift, load_features_from_redshift

from vault import get_vault_secrets


def segment_classifier_predictions(df, clf_predicted_probability_column):
    # create segments out of classifier output (probability prediction)
    conditions = [
        df[clf_predicted_probability_column].between(0, 0.5),
        df[clf_predicted_probability_column].between(0.5, 0.75),
        df[clf_predicted_probability_column].between(0.75, 1)]
    choices = ["tier_3", "tier_2", "tier_1"]
    segment_col = "tier"
    df[segment_col] = np.select(conditions, choices, default=np.nan)

    # sort
    df = df.sort_values([clf_predicted_probability_column], ascending=[False])

    return df


def segment_classifier_and_regressor_predictions(df,
                                                 first_clf_predicted_probability_column,
                                                 first_reg_predicted_column=None,
                                                 second_clf_predicted_probability_column=None,
                                                 second_reg_predicted_column=None,
                                                 is_multiple_pipeline=False
                                                 ):
    # create segments out of classifier and regressor output
    if is_multiple_pipeline:
        conditions = \
            [
                (
                    (df[first_clf_predicted_probability_column] < 0.5)

                ),
                (
                        (df[first_clf_predicted_probability_column] >= 0.5) &
                        (df[second_clf_predicted_probability_column] < 0.5)
                ),
                (
                        (df[first_clf_predicted_probability_column] >= 0.5) &
                        (df[second_clf_predicted_probability_column] >= 0.5)
                )
            ]
        choices = ["tier_3", "tier_2", "tier_1"]
        segment_col = "tier"
        df[segment_col] = np.select(conditions, choices, default=np.nan)

        # sort
        df = df.sort_values([first_clf_predicted_probability_column], ascending=[False])

    return df


def classifier_predictions_pipeline(df,
                                    add_features_from_redshift_to_df_input,
                                    models_name,
                                    model_types,
                                    clf_config
                                    ):
    # retrieve pipeline from config file
    pipeline = clf_config["Pipeline"].get("pipeline")
    pipeline_name = clf_config["Pipeline"].get("pipeline_name")

    # retrieve columns_to_be_stored_pre_processing and prediction_column from config file
    clf_cols_to_store_pre_processing = ast.literal_eval(
        clf_config["Data_Processing"].get("cols_to_store_pre_processing"))
    clf_prediction_column = ast.literal_eval(clf_config["Data_Processing"].get("prediction_column"))

    # retrieve model_file_location from config file
    model_file_location = clf_config["Locations"].get("model_file_location")

    # if df == input csv - then add to it features from redshift
    if add_features_from_redshift_to_df_input:
        # retrieve redshift credentials
        if model_file_location == "GCS":
            credentials = get_vault_secrets()[0]
        elif model_file_location == "LOCAL" or model_file_location == "GCS_VIA_LOCAL":
            credentials = None
        else:
            raise ValueError("No location was specified in the config file")
        # connect redshift
        cur = connect_redshift(credentials=credentials)[2]
        # add features to csv file
        df = load_features_from_redshift(df=df,
                                         cur=cur,
                                         config=clf_config
                                         )

    # initialize prediction process
    print("-------------------------------------------------------------")
    print(
        f"Prediction process for pipeline {', '.join(models_name)} ({', '.join(model_types)}) - started at: \033[1m{datetime.now()}\033[0m")

    # -------------------------- CLASSIFIER PIPELINE -------------------------- #
    # load clf model
    trained_clf_model_object, trained_clf_model_cols = load_model_pkl_file(clf_config)

    # store columns pre data processing
    stored_columns = df[clf_cols_to_store_pre_processing]

    # -------------------------- ACCEPTANCE RATE PIPELINE -------------------------- #
    # -------------------------- RAID TUTORIALS PIPELINE --------------------------- #
    # -------------------------- YOUTUBE CONVERSIONS PIPELINE ---------------------- #
    if pipeline_name == "RAID_Tutorials" or pipeline_name == "acceptance_rate" or pipeline_name == "YouTube_Conversions":
        # retrieve model bucketed columns (similar to columns from training dataset)
        model_bucketed_columns = retrieve_model_bucketed_columns(trained_clf_model_cols, clf_config)
        # bucket categorical features of the df (csv file)
        df = df_apply_bucket_transformation(df, model_bucketed_columns, clf_config)
        # prepare data for the trained model
        df = prepare_data_for_trained_model(df=df,
                                            cols_when_model_builds=trained_clf_model_cols,
                                            is_drop_cols=True,
                                            is_convert_numeric_cols_to_categorical=True,
                                            is_get_dummies=True,
                                            is_add_zeros_to_missing_cols=True,
                                            is_reorder_df_cols_to_match_trained_model=True,
                                            config=clf_config
                                            )
        # predict with the model
        df[clf_prediction_column] = trained_clf_model_object.predict_proba(df).T[1]
        # calibrate predictions
        df = probability_calibration(df, clf_config)
        # prepare final df
        df_predictions = df.join(stored_columns)
        df_predictions = df_predictions[[clf_cols_to_store_pre_processing,
                                         clf_prediction_column]]

    # -------------------------- RAID DEPOSITS PIPELINE -------------------------- #
    elif pipeline_name == "RAID_Deposits":
        # prepare data for the trained model
        df = prepare_data_for_trained_model(df=df,
                                            cols_when_model_builds=trained_clf_model_cols,
                                            is_drop_cols=True,
                                            is_convert_numeric_cols_to_categorical=False,
                                            is_get_dummies=False,
                                            is_add_zeros_to_missing_cols=False,
                                            is_reorder_df_cols_to_match_trained_model=True,
                                            config=clf_config
                                            )
        # predict with the model
        df[clf_prediction_column] = trained_clf_model_object.predict_proba(df).T[1]
        # calibrate predictions
        df = probability_calibration(df, clf_config)
        # prepare final df
        df_predictions = df.join(stored_columns)
        df_predictions = df_predictions[[clf_cols_to_store_pre_processing[0],
                                         clf_cols_to_store_pre_processing[1],
                                         clf_prediction_column, 'tutorials_d7']]

    else:
        raise ValueError("\033[1m pipeline_name was not defined correctly in the config file \033[0m")

    print(
        f"Prediction process for pipeline {', '.join(models_name)} ({', '.join(model_types)}) - finished at: \033[1m{datetime.now()}\033[0m")
    print("-------------------------------------------------------------")

    return df_predictions, pipeline, pipeline_name, clf_prediction_column, clf_cols_to_store_pre_processing


def regressor_predictions_pipeline(df,
                                   add_features_from_redshift_to_df_input,
                                   models_name,
                                   model_types,
                                   reg_config
                                   ):
    # retrieve pipeline from config file
    pipeline = reg_config["Pipeline"].get("pipeline")
    pipeline_name = reg_config["Pipeline"].get("pipeline_name")

    # retrieve columns_to_be_stored_pre_processing and prediction_column from config file
    reg_cols_to_store_pre_processing = ast.literal_eval(
        reg_config["Data_Processing"].get("cols_to_store_pre_processing"))
    reg_prediction_column = ast.literal_eval(reg_config["Data_Processing"].get("prediction_column"))

    # retrieve model_file_location from config file
    model_file_location = reg_config["Locations"].get("model_file_location")

    # if df == input csv - then add to it features from redshift
    if add_features_from_redshift_to_df_input:
        # retrieve redshift credentials
        if model_file_location == "GCS":
            credentials = get_vault_secrets()[0]
        elif model_file_location == "LOCAL" or model_file_location == "GCS_VIA_LOCAL":
            credentials = None
        else:
            raise ValueError("No location was specified in the config file")
        # connect redshift
        cur = connect_redshift(credentials=credentials)[2]
        # add features to csv file
        df = load_features_from_redshift(df=df,
                                         cur=cur,
                                         config=reg_config
                                         )

    # initialize prediction process
    print("-------------------------------------------------------------")
    print(
        f"Prediction process for pipeline {', '.join(models_name)} ({', '.join(model_types)}) - started at: \033[1m{datetime.now()}\033[0m")

    # -------------------------- REGRESSOR PIPELINE -------------------------- #
    # load regressor model
    trained_reg_model_object, trained_reg_model_cols = load_model_pkl_file(reg_config)
    # store columns pre data processing
    stored_columns = df[reg_cols_to_store_pre_processing]
    # prepare data for the trained model
    df = prepare_data_for_trained_model(df=df,
                                        cols_when_model_builds=trained_reg_model_cols,
                                        is_drop_cols=True,
                                        is_convert_numeric_cols_to_categorical=True,
                                        is_get_dummies=True,
                                        is_add_zeros_to_missing_cols=True,
                                        is_reorder_df_cols_to_match_trained_model=True,
                                        config=reg_config
                                        )
    # predict with the regressor model
    df[reg_prediction_column] = trained_reg_model_object.predict(df)
    # prepare final df
    df_predictions = df.join(stored_columns)

    print(
        f"Prediction process for pipeline {', '.join(models_name)} ({', '.join(model_types)}) - finished at: \033[1m{datetime.now()}\033[0m")
    print("-------------------------------------------------------------")

    return df_predictions, pipeline, pipeline_name, reg_prediction_column, reg_cols_to_store_pre_processing


def classifier_and_regressor_predictions_pipeline(df,
                                                  add_features_from_redshift_to_df_input,
                                                  models_name,
                                                  model_types,
                                                  clf_config,
                                                  reg_config
                                                  ):
    # retrieve pipeline from config file (either from clf_config OR reg_config)
    pipeline = reg_config["Pipeline"].get("pipeline")
    pipeline_name = reg_config["Pipeline"].get("pipeline_name")

    # retrieve columns_to_be_stored_pre_processing and prediction_column from config file
    clf_cols_to_store_pre_processing = ast.literal_eval(
        clf_config["Data_Processing"].get("cols_to_store_pre_processing"))
    clf_prediction_column = ast.literal_eval(clf_config["Data_Processing"].get("prediction_column"))
    reg_cols_to_store_pre_processing = ast.literal_eval(
        reg_config["Data_Processing"].get("cols_to_store_pre_processing"))
    reg_prediction_column = ast.literal_eval(reg_config["Data_Processing"].get("prediction_column"))

    # retrieve model_file_location from config file
    model_file_location = clf_config["Locations"].get("model_file_location")

    # if df == input csv then add to it features from redshift
    if add_features_from_redshift_to_df_input:
        # retrieve redshift credentials (from clf_config ONLY)
        if model_file_location == "GCS":
            credentials = get_vault_secrets()[0]
        elif model_file_location == "LOCAL" or model_file_location == "GCS_VIA_LOCAL":
            credentials = None
        else:
            raise ValueError("No location was specified in the config file")
        # connect redshift (from clf_config ONLY)
        cur = connect_redshift(credentials=credentials)[2]
        # add features to csv file
        df = load_features_from_redshift(df=df,
                                         cur=cur,
                                         config=clf_config
                                         )

    # initialize prediction process
    print("-------------------------------------------------------------")
    print(
        f"Prediction process for pipeline {', '.join(models_name)} ({', '.join(model_types)}) - started at: \033[1m{datetime.now()}\033[0m")

    # -------------------------- CLASSIFIER PIPELINE -------------------------- #
    # load classifier model
    trained_clf_model_object, trained_clf_model_cols = load_model_pkl_file(clf_config)
    # store columns pre data processing
    stored_columns = df[clf_cols_to_store_pre_processing]
    # process df according to the relevant pipeline
    if pipeline_name == "RAID_Tutorials":
        # retrieve classifier model bucketed columns (similar to columns from training dataset)
        model_bucketed_columns = retrieve_model_bucketed_columns(trained_clf_model_cols, clf_config)
        # bucket categorical features of the df_input (csv file)
        df = df_apply_bucket_transformation(df, model_bucketed_columns, clf_config)
        # prepare data for the trained classifier model
        df = prepare_data_for_trained_model(df=df,
                                            cols_when_model_builds=trained_clf_model_cols,
                                            is_drop_cols=True,
                                            is_convert_numeric_cols_to_categorical=True,
                                            is_get_dummies=True,
                                            is_add_zeros_to_missing_cols=True,
                                            is_reorder_df_cols_to_match_trained_model=True,
                                            config=clf_config
                                            )
    elif pipeline_name == "RAID_Deposits":
        # prepare data for the trained classifier model
        df = prepare_data_for_trained_model(df=df,
                                            cols_when_model_builds=trained_clf_model_cols,
                                            is_drop_cols=True,
                                            is_convert_numeric_cols_to_categorical=False,
                                            is_get_dummies=False,
                                            is_add_zeros_to_missing_cols=False,
                                            is_reorder_df_cols_to_match_trained_model=True,
                                            config=clf_config
                                            )
    # predict with the classifier model
    df[clf_prediction_column] = trained_clf_model_object.predict_proba(df).T[1]
    # calibrate predictions
    df = probability_calibration(df, clf_config)
    # prepare final df for the regressor
    df = df.join(stored_columns)

    # -------------------------- REGRESSOR PIPELINE -------------------------- #
    # load regressor model
    trained_reg_model_object, trained_reg_model_cols = load_model_pkl_file(reg_config)
    # store columns pre data processing
    stored_columns = df[reg_cols_to_store_pre_processing]
    # prepare data for the trained regressor model
    df = prepare_data_for_trained_model(df=df,
                                        cols_when_model_builds=trained_reg_model_cols,
                                        is_drop_cols=True,
                                        is_convert_numeric_cols_to_categorical=False,
                                        is_get_dummies=False,
                                        is_add_zeros_to_missing_cols=False,
                                        is_reorder_df_cols_to_match_trained_model=True,
                                        config=reg_config
                                        )
    # predict with the regressor model
    df[reg_prediction_column] = trained_reg_model_object.predict(df)
    # prepare final df
    df_predictions = df.join(stored_columns)

    print(
        f"Prediction process for pipeline {', '.join(models_name)} ({', '.join(model_types)}) - finished at: \033[1m{datetime.now()}\033[0m")
    print("-------------------------------------------------------------")

    return df_predictions, pipeline, pipeline_name, reg_prediction_column, clf_prediction_column, clf_cols_to_store_pre_processing


def ml_predictions_pipeline(df: pd.DataFrame,
                            add_features_from_redshift_to_df_input: bool,
                            models_name: list,
                            model_types: list,
                            is_previous_pipeline_clf_reg=None,
                            config_objects=None
                            ):
    # retrieve config
    for config in config_objects.values():

        if len(model_types) == 1 and model_types == ["clf"]:
            # retrieve config file
            clf_config = config[0]
            # initialize classifier pipeline
            if clf_config["Pipeline"].get("pipeline") == "CLASSIFIER":
                df_predictions, pipeline, pipeline_name, clf_prediction_column, clf_cols_to_store_pre_processing \
                    = classifier_predictions_pipeline(df,
                                                      add_features_from_redshift_to_df_input,
                                                      models_name,
                                                      model_types,
                                                      clf_config
                                                      )
            else:
                raise ValueError("\033[1m Pipeline object was not defined correctly in the config file \033[0m")

        elif len(model_types) == 1 and model_types == ["reg"]:
            # retrieve config file
            reg_config = config[0]
            # initialize regressor pipeline
            if reg_config["Pipeline"].get("pipeline") == "REGRESSOR":
                df_predictions, pipeline, pipeline_name, reg_prediction_column, reg_cols_to_store_pre_processing \
                    = regressor_predictions_pipeline(df,
                                                     add_features_from_redshift_to_df_input,
                                                     models_name,
                                                     model_types,
                                                     reg_config
                                                     )
            else:
                raise ValueError("\033[1m Pipeline object was not defined correctly in the config file \033[0m")

        elif len(model_types) == 2 and model_types == ["clf", "reg"]:
            # retrieve config file
            clf_config = config[0]
            reg_config = config[1]
            # initialize classifier and regressor pipeline
            if clf_config["Pipeline"].get("pipeline") == "CLASSIFIER+REGRESSOR":
                # check that the configs are the same
                if (clf_config["Pipeline"].get("pipeline") !=
                    reg_config["Pipeline"].get("pipeline")) \
                        or \
                        (clf_config["Pipeline"].get("pipeline_name") !=
                         reg_config["Pipeline"].get("pipeline_name")) \
                        or \
                        (clf_config["Locations"].get("model_file_location")
                         != reg_config["Locations"].get("model_file_location")):
                    raise ValueError(
                        "\033[1m A conflict was found between clf_config and reg_config files regarding CLASSIFIER+REGRESSOR pipeline \033[0m")
                else:
                    df_predictions, pipeline, pipeline_name, reg_prediction_column, clf_prediction_column, clf_cols_to_store_pre_processing \
                        = classifier_and_regressor_predictions_pipeline(df,
                                                                        add_features_from_redshift_to_df_input,
                                                                        models_name,
                                                                        model_types,
                                                                        clf_config,
                                                                        reg_config
                                                                        )
            else:
                raise ValueError("\033[1m Pipeline object was not defined correctly in the config file \033[0m")

        else:
            raise ValueError("\033[1m models_name and model_types were not defined correctly \033[0m")

    # finalize df before return

    # -------------------------- ACCEPTANCE RATE PIPELINE -------------------------- #
    if pipeline_name == "acceptance_rate":
        df_predictions = \
            segment_classifier_predictions(df=df_predictions,
                                           clf_predicted_probability_column=clf_prediction_column
                                           )
        return df_predictions, clf_config

    # -------------------------- YOUTUBE CONVERSIONS PIPELINE ---------------------- #
    elif pipeline_name == "YouTube_Conversions":
        df_predictions = \
            segment_classifier_predictions(df=df_predictions,
                                           clf_predicted_probability_column=clf_prediction_column
                                           )
        return df_predictions, clf_config

    # -------------------------- RAID TUTORIALS PIPELINE -------------------------- #
    elif pipeline_name == "RAID_Tutorials":
        if pipeline == "CLASSIFIER+REGRESSOR":
            is_previous_pipeline_clf_reg = True
            return df_predictions, is_previous_pipeline_clf_reg, clf_config
        else:
            is_previous_pipeline_clf_reg = False
            return df_predictions, is_previous_pipeline_clf_reg, clf_config

    # -------------------------- RAID DEPOSITS PIPELINE -------------------------- #
    elif pipeline_name == "RAID_Deposits":

        if pipeline == "CLASSIFIER+REGRESSOR":
            df_predictions = \
                df_predictions[[clf_cols_to_store_pre_processing[0],
                                clf_cols_to_store_pre_processing[1],
                                'tutorials_d7',
                                clf_prediction_column,
                                reg_prediction_column]]
            df_predictions = \
                segment_classifier_and_regressor_predictions(
                    df=df_predictions,
                    first_clf_predicted_probability_column=clf_cols_to_store_pre_processing[1],
                    first_reg_predicted_column='tutorials_d7',
                    second_clf_predicted_probability_column=clf_prediction_column,
                    second_reg_predicted_column=reg_prediction_column,
                    is_multiple_pipeline=True
                )

        elif pipeline == "CLASSIFIER":
            if is_previous_pipeline_clf_reg:
                df_predictions = \
                    df_predictions[[clf_cols_to_store_pre_processing[0],
                                    clf_cols_to_store_pre_processing[1],
                                    clf_prediction_column,
                                    'tutorials_d7']]
                df_predictions = \
                    segment_classifier_and_regressor_predictions(
                        df=df_predictions,
                        first_clf_predicted_probability_column=clf_cols_to_store_pre_processing[1],
                        second_clf_predicted_probability_column=clf_prediction_column,
                        is_multiple_pipeline=True
                    )

            else:
                df_predictions = \
                    df_predictions[[clf_cols_to_store_pre_processing[0],
                                    clf_cols_to_store_pre_processing[1],
                                    'tutorials_d7',
                                    reg_prediction_column]]
                df_predictions = \
                    segment_classifier_predictions(df=df_predictions,
                                                   clf_predicted_probability_column=reg_cols_to_store_pre_processing[1]
                                                   )

        elif pipeline == "REGRESSOR":
            df_predictions = \
                df_predictions[[reg_cols_to_store_pre_processing[0],
                                reg_cols_to_store_pre_processing[1],
                                'tutorials_d7',
                                reg_prediction_column]]
            df_predictions = \
                segment_classifier_predictions(df=df_predictions,
                                               clf_predicted_probability_column=reg_cols_to_store_pre_processing[1]
                                               )

        else:
            raise ValueError("\033[1m Pipeline object was not defined correctly in the config file \033[0m")

        return df_predictions

    else:
        raise ValueError("\033[1m pipeline_name was not defined correctly in the config file \033[0m")


def create_acceptance_rate_predictions_df(df_input,
                                          add_features_from_redshift_to_df_input
                                          ):
    # define model names/types to predict with
    models_name = ["acceptance_rate"]
    model_types = ["clf"]

    # load config file
    config_objects = load_config(models_name, model_types)

    # initialize acceptance_rate pipeline
    df_predictions, acceptance_rate_clf_config = \
        ml_predictions_pipeline(df=df_input,
                                add_features_from_redshift_to_df_input=add_features_from_redshift_to_df_input,
                                models_name=models_name,
                                model_types=model_types,
                                config_objects=config_objects
                                )
    return df_predictions, acceptance_rate_clf_config


def create_tutorials_predictions_df(df_input,
                                    add_features_from_redshift_to_df_input
                                    ):
    # define model names/types to predict with
    models_name = ["raid_tutorials"]
    model_types = ["clf", "reg"]  # model_types = ["clf", "reg"]

    # load config files
    config_objects = load_config(models_name, model_types)

    # initialize raid_tutorials pipeline
    df_tutorials_predictions, is_previous_pipeline_clf_reg, tutorials_clf_config = \
        ml_predictions_pipeline(df=df_input,
                                add_features_from_redshift_to_df_input=add_features_from_redshift_to_df_input,
                                models_name=models_name,
                                model_types=model_types,
                                config_objects=config_objects
                                )
    return df_tutorials_predictions, is_previous_pipeline_clf_reg, tutorials_clf_config


def create_deposits_predictions_df(df_input,
                                   add_features_from_redshift_to_df_input,
                                   is_previous_pipeline_clf_reg
                                   ):
    # define model names/types to predict with
    models_name = ["raid_deposits"]
    model_types = ["clf"]  # model_types = ["clf", "reg"]

    # load config files
    config_objects = load_config(models_name, model_types)

    # initialize raid_deposits pipeline
    df_tutorials_and_deposits_predictions = \
        ml_predictions_pipeline(df=df_input,
                                add_features_from_redshift_to_df_input=add_features_from_redshift_to_df_input,
                                models_name=models_name,
                                model_types=model_types,
                                is_previous_pipeline_clf_reg=is_previous_pipeline_clf_reg,
                                config_objects=config_objects
                                )

    return df_tutorials_and_deposits_predictions


def create_youtube_conversions_predictions_df(df_input,
                                              add_features_from_redshift_to_df_input
                                              ):
    # define model names/types to predict with
    models_name = ["youtube_conversions"]
    model_types = ["clf"]

    # load config file
    config_objects = load_config(models_name, model_types)

    # initialize acceptance_rate pipeline
    df_predictions, youtube_conversions_clf_config = \
        ml_predictions_pipeline(df=df_input,
                                add_features_from_redshift_to_df_input=add_features_from_redshift_to_df_input,
                                models_name=models_name,
                                model_types=model_types,
                                config_objects=config_objects
                                )
    return df_predictions, youtube_conversions_clf_config
