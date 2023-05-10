from datetime import datetime, timezone
from google.cloud import storage
from google.auth import compute_engine
import pandas as pd
import uuid

from prediction_projects.model_utils import connect_redshift
from prediction_projects.predict import \
    create_tutorials_predictions_df, create_deposits_predictions_df, \
    create_acceptance_rate_predictions_df, create_youtube_conversions_predictions_df

from ml_projects_utilities.sql_utils import read_sql_file, actual_create_table, get_table_summary

from sqlalchemy.engine import Connection


def write_csv_to_local_file(df, object_name):
    file_name = object_name + "_" + str(datetime.now().strftime("%Y%m%d_%H%M")) + ".csv"
    df.to_csv(file_name, index=False)


def prepare_df_for_json_file(df, database_name, collection_name):
    df["_metadata"] = df.apply(
        lambda row: {"timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                     "database_name": database_name,
                     "collection_name": collection_name,
                     "op": "i",
                     "doc_id ": str(uuid.uuid4())
                     },
        axis=1
    )
    df["data"] = df.apply(lambda row: {"boss_channel_id": row["boss_channel_id"],
                                       "tier": row["tier"]},
                          axis=1
                          )
    df = df[["_metadata", "data"]]

    return df


def write_json_to_local_file(df, object_name):
    file_name = str(object_name) + "_" + str(datetime.now().strftime("%Y%m%d_%H%M")) + ".json"
    with open(file_name, 'w') as f:
        f.write(df.to_json(orient='records', lines=True))


def upload_to_gcs(file_name, drop_if_exists, config):
    location_for_writing_predictions = config["Locations"].get("location_for_writing_predictions")

    if location_for_writing_predictions == "GCS" or location_for_writing_predictions == "GCS_VIA_LOCAL":
        # configure paths
        gcs_bucket = config["GCS"].get("bucket")
        gcs_folder_path = config["GCS"].get("folder_path_of_predictions_file")
        gcs_file_path = str(gcs_folder_path) + str(file_name)
        # configure environment
        if location_for_writing_predictions == "GCS":
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
            if drop_if_exists:
                # delete file
                blob = bucket.blob(gcs_file_path)
                blob.delete()
                # upload file
                blob = bucket.blob(gcs_file_path)
                blob.upload_from_filename(file_name)
            else:
                pass
        else:
            # upload file
            blob = bucket.blob(gcs_file_path)
            blob.upload_from_filename(file_name)

    else:
        raise ValueError("No location was specified in the config file")


def write_json_to_gcs_bucket(df, object_name, drop_if_exists, config):
    # create predictions json file
    file_name = str(object_name) + "_" + str(datetime.now().strftime("%Y%m%d_%H%M")) + ".json"
    df.to_json(file_name, orient='records', lines=True)
    # upload file to GCS
    upload_to_gcs(file_name=file_name, drop_if_exists=drop_if_exists, config=config)


def write_predictions(engine,
                      conn,
                      cur,
                      schema_name,
                      df,
                      object_name,
                      database_name,
                      skip_if_exists,
                      drop_if_exists,
                      create_table_from_sql_or_df,
                      config
                      ):
    location_for_writing_predictions = config["Locations"].get("location_for_writing_predictions")

    print(f"Writing predictions to \033[1m{location_for_writing_predictions}\033[0m")

    if location_for_writing_predictions == "LOCAL_CSV":
        write_csv_to_local_file(df=df, object_name=object_name)

    elif location_for_writing_predictions == "LOCAL_JSON":
        df = prepare_df_for_json_file(df=df,
                                      database_name=database_name,
                                      collection_name=object_name)
        write_json_to_local_file(df=df,
                                 object_name=object_name)

    elif location_for_writing_predictions == "REDSHIFT":
        actual_create_table(engine=engine,
                            conn=conn,
                            cur=cur,
                            schema=schema_name,
                            table_name=object_name,
                            create_table_from_sql_or_df=create_table_from_sql_or_df,
                            sql=read_sql_file(object_name),
                            df=df,
                            skip_if_exists=skip_if_exists,
                            drop_if_exists=drop_if_exists
                            )

    elif location_for_writing_predictions == "GCS" or location_for_writing_predictions == "GCS_VIA_LOCAL":
        df = prepare_df_for_json_file(df=df,
                                      database_name=database_name,
                                      collection_name=object_name)
        write_json_to_gcs_bucket(df=df,
                                 object_name=object_name,
                                 drop_if_exists=drop_if_exists,
                                 config=config
                                 )

    else:
        raise ValueError("No location_for_writing_predictions was specified in the config file")

    print("-------------------------------------------------------------")


def write_kronos_predictions(engine: Connection,
                             conn: Connection,
                             cur: Connection,
                             schema_name: str,
                             do_the_job: bool = True,
                             skip_if_exists=False,
                             drop_if_exists: bool = False,
                             create_table_from_sql_or_df: str = None
                             ) -> str:
    """ Kronos (Twitch QoD) predictions """

    object_name = "twitch_kronos_predictions"

    if do_the_job:
        print("-------------------------------------------------------------")
        print(f"Process write_kronos_predictions - started at: \033[1m{datetime.now()}\033[0m")
        print("-------------------------------------------------------------")

        # fetch features data from sql query
        print(f"Loading data from REDSHIFT - started at: \033[1m{datetime.now()}\033[0m")
        sql = read_sql_file(object_name)
        cur.execute(sql)
        print(f"Loading data from REDSHIFT - finished at: \033[1m{datetime.now()}\033[0m")

        # convert features data into df
        df_query = pd.DataFrame(cur.fetchall())
        df_query.columns = [desc[0] for desc in cur.description]

        # drop duplications
        df_query = df_query.drop_duplicates()

        # create df_tutorials_predictions
        df_tutorials_predictions, is_previous_pipeline_clf_reg, tutorials_clf_config = \
            create_tutorials_predictions_df(df_input=df_query,
                                            add_features_from_redshift_to_df_input=False
                                            )

        # create df_tutorials_and_deposits_predictions
        df_tutorials_and_deposits_predictions = \
            create_deposits_predictions_df(df_input=df_tutorials_predictions,
                                           add_features_from_redshift_to_df_input=False,
                                           is_previous_pipeline_clf_reg=is_previous_pipeline_clf_reg
                                           )

        # write predictions
        write_predictions(engine=engine,
                          conn=conn,
                          cur=cur,
                          schema_name=schema_name,
                          df=df_tutorials_and_deposits_predictions,
                          object_name=object_name,
                          database_name="kronos",
                          skip_if_exists=skip_if_exists,
                          drop_if_exists=drop_if_exists,
                          create_table_from_sql_or_df=create_table_from_sql_or_df,
                          config=tutorials_clf_config
                          )

        print(f"Process write_kronos_predictions - finished at: \033[1m{datetime.now()}\033[0m")
        print("-------------------------------------------------------------")

        return object_name


def write_acceptance_rate_predictions(engine: Connection,
                                      conn: Connection,
                                      cur: Connection,
                                      schema_name: str,
                                      do_the_job: bool = False,
                                      skip_if_exists=False,
                                      drop_if_exists: bool = False,
                                      create_table_from_sql_or_df: str = None
                                      ) -> str:
    """ Acceptance Rate predictions """

    object_name = "twitch_acceptance_rate_predictions"

    if do_the_job:
        print("-------------------------------------------------------------")
        print(f"Process write_acceptance_rate_predictions - started at: \033[1m{datetime.now()}\033[0m")
        print("-------------------------------------------------------------")

        # fetch features data from sql query
        print(f"Loading data from REDSHIFT - started at: \033[1m{datetime.now()}\033[0m")
        sql = read_sql_file(object_name)
        cur.execute(sql)
        print(f"Loading data from REDSHIFT - finished at: \033[1m{datetime.now()}\033[0m")

        # convert features data into df
        df_query = pd.DataFrame(cur.fetchall())
        df_query.columns = [desc[0] for desc in cur.description]

        # drop duplications
        df_query = df_query.drop_duplicates()

        # create df_acceptance_rate_predictions
        df_acceptance_rate_predictions, acceptance_rate_clf_config = \
            create_acceptance_rate_predictions_df(df_input=df_query,
                                                  add_features_from_redshift_to_df_input=False
                                                  )

        # write predictions
        write_predictions(engine=engine,
                          conn=conn,
                          cur=cur,
                          schema_name=schema_name,
                          df=df_acceptance_rate_predictions,
                          object_name=object_name,
                          database_name=object_name,
                          skip_if_exists=skip_if_exists,
                          drop_if_exists=drop_if_exists,
                          create_table_from_sql_or_df=create_table_from_sql_or_df,
                          config=acceptance_rate_clf_config
                          )

        print(f"Process write_acceptance_rate_predictions - finished at: \033[1m{datetime.now()}\033[0m")
        print("-------------------------------------------------------------")

        return object_name


def write_youtube_conversions_predictions(engine: Connection,
                                          conn: Connection,
                                          cur: Connection,
                                          schema_name: str,
                                          do_the_job: bool = False,
                                          skip_if_exists=False,
                                          drop_if_exists: bool = False,
                                          create_table_from_sql_or_df: str = None
                                          ) -> str:
    """ Rhea (YouTube Conversions) predictions """

    object_name = "youtube_conversions_predictions"

    if do_the_job:
        print("-------------------------------------------------------------")
        print(f"Process write_youtube_conversions_predictions - started at: \033[1m{datetime.now()}\033[0m")
        print("-------------------------------------------------------------")

        # fetch features data from sql query
        print(f"Loading data from REDSHIFT - started at: \033[1m{datetime.now()}\033[0m")
        sql = read_sql_file(object_name)
        cur.execute(sql)
        print(f"Loading data from REDSHIFT - finished at: \033[1m{datetime.now()}\033[0m")

        # convert features data into df
        df_query = pd.DataFrame(cur.fetchall())
        df_query.columns = [desc[0] for desc in cur.description]

        # drop duplications
        df_query = df_query.drop_duplicates()

        # create df_acceptance_rate_predictions
        df_youtube_conversions_predictions, youtube_conversions_clf_config = \
            create_youtube_conversions_predictions_df(df_input=df_query,
                                                      add_features_from_redshift_to_df_input=False
                                                      )

        # write predictions
        write_predictions(engine=engine,
                          conn=conn,
                          cur=cur,
                          schema_name=schema_name,
                          df=df_youtube_conversions_predictions,
                          object_name=object_name,
                          database_name="rhea",
                          skip_if_exists=skip_if_exists,
                          drop_if_exists=drop_if_exists,
                          create_table_from_sql_or_df=create_table_from_sql_or_df,
                          config=youtube_conversions_clf_config
                          )

        print(f"Process write_youtube_conversions_predictions - finished at: \033[1m{datetime.now()}\033[0m")
        print("-------------------------------------------------------------")

        return object_name


def main(*,
         schema_interim: str = "dev_yehuda",
         create_acceptance_rate_predictions: bool = False,
         create_kronos_predictions: bool = True,
         create_youtube_conversions_predictions: bool = False,
         create_all: bool = False,
         force_creation_if_exists: bool = True,
         verbose=False
         ):
    """
    Description:
    This script is used to write the predictions output, either to:
        * LOCAL_CSV - Write the predictions to a local csv file.
        * LOCAL_JSON - Write the predictions to a local json file.
        * REDSHIFT - Generate predictions tables. All the tables are generated in Redshift under the same schema.
                     The current schema is `dev_yehuda`, but you can specify your own by using the `schema_interim` parameter.
        * GCS - Write the predictions to GCS bucket, using a script that runs in the cloud.
        * GCS_VIA_LOCAL - Write the predictions to GCS bucket, using a script that runs locally.

    Parameters:
    :param schema_interim:
        The target schema where we save the interim data
    :param create_acceptance_rate_predictions:
        Should we create acceptance rate predictions
    :param create_kronos_predictions:
        Should we create kronos predictions
    :param create_youtube_conversions_predictions:
        Should we create YouTube conversions predictions
    :param create_all:
        Should we create all the predictions. If this parameter is True, then all the other parameters are.
    :param force_creation_if_exists:
        Should we force the creation of the predictions if they already exist.
    :param verbose:
        Should we print created predictions summary
    """

    # create all predictions
    if create_all:
        create_acceptance_rate_predictions = True
        create_kronos_predictions = True
        create_youtube_conversions_predictions = True

    # connect redshift
    engine, conn, cur = connect_redshift(credentials=None)

    # write predictions
    written_predications_objects = \
        [
            write_acceptance_rate_predictions(engine=engine,
                                              conn=conn,
                                              cur=cur,
                                              schema_name=schema_interim,
                                              do_the_job=create_acceptance_rate_predictions,
                                              drop_if_exists=force_creation_if_exists,
                                              create_table_from_sql_or_df="df"  # choose between "df" / "sql"
                                              ),
            write_kronos_predictions(engine=engine,
                                     conn=conn,
                                     cur=cur,
                                     schema_name=schema_interim,
                                     do_the_job=create_kronos_predictions,
                                     drop_if_exists=force_creation_if_exists,
                                     create_table_from_sql_or_df="df"  # choose between "df" / "sql"
                                     ),
            write_youtube_conversions_predictions(engine=engine,
                                                  conn=conn,
                                                  cur=cur,
                                                  schema_name=schema_interim,
                                                  do_the_job=create_youtube_conversions_predictions,
                                                  drop_if_exists=force_creation_if_exists,
                                                  create_table_from_sql_or_df="df"  # choose between "df" / "sql"
                                                  )
        ]

    if verbose:
        print("Uploaded the following files:")
        for f in written_predications_objects:
            print(f)
            print("\n\n")


if __name__ == "__main__":
    main()
