
# SE ML Applications

Applications of several projects, the main entry points are: ```main_flask.py```, ```main_writing_predictions.py```.

## Author

- [@B-Yehuda](https://github.com/B-Yehuda)


## Environment Setup

All Python packages necessary to run the code in this repository are listed in `requirements.txt`. To create a new Anaconda environment which includes these packages, enter the following command in your terminal:

```bash
conda create --name ml_apps --file requirements.txt
conda activate ml_apps
```


## Code Execution
You will need either Redshift or YouTube API credentials to run the code. Put the credentials in ```data/secrets/redshift_config.json``` or ```data/secrets/youtube_config.json``` respectively. **Make sure not to commit those files to the repository**.

To execute the applications execute one of the following commands:
```bash
main_flask.py
main_writing_predictions.py
```


## Running unit tests

**TODO**: To run the unit tests, simply run pytest in the test_src directory. 

```bash
cd test_src
pytest
```


## Project Tree
```bash
├── main_flask.py   
├── main_writing_predictions.py   

├── .github/workflows
  ├── docker-build-push.yml

├── configs
  ├── config_clf_acceptance_rate.ini
  ├── config_clf_raid_tutorials.ini
  ├── config_reg_raid_tutorials.ini
  ├── config_clf_raid_deposits.ini
  ├── config_reg_raid_deposits.ini
  ├── config_clf_youtube_conversions.ini
  ├── config_youtube_search.INI

├── data
  ├── sql
    ├── twitch_acceptance_rate_predictions.sql
    ├── twitch_kronos_predictions.sql
    ├── youtube_conversions_predictions.sql

├── ml_projects_utilities 
    ├── __init__.py 
    ├── load_config_utils.py
    ├── sql_utils.py

├── prediction_projects 
    ├── __init__.py 
    ├── model_utils.py
    ├── predict.py

├── static
    ├── MERCURY.jpg
    ├── RAID.jpg
    ├── SE.jpg

├── templates 
    ├── index_acceptance_rate.html
    ├── index_home.html
    ├── index_kronos.html
    ├── index_youtube_search.html

├── youtube_lookalikes_project
    ├── __init__.py 
    ├── youtube_scraping_functions.py
    ├── youtube_search.py

├── vault.py

├── .gitignore

├── Dockerfile

├── requirements.txt

├── README.md
```
