# Disaster Response Pipeline Project




https://user-images.githubusercontent.com/78309614/161602346-0ec13dfd-e1c4-4de0-b26a-7d6d8eda040b.mp4




## Table of contents:

1. [Installation](#installation)
2. [Project Description](#description)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

run `pip install -r requirements.txt` to install all required libs.

The code should run with no issues using Python versions 3.*.


## Project Description <a name="description"></a>

This projects aims to build a model for an API that classifies disaster messages, so that they can be sent to the appropriate disaster relief agency. 

When a disaster happens is when the agencies are flooded with messages and it's also when they have the least capacity to deal with it.

The goal is to speed up the process of recognizing important messages and redirecting them correctly.

This project is composed of the following steps:

1. Take real data from tweets and text messages sent during real life disaster events;
2. Prepare this data with an ETL Pipeline;
3. Build a Machine Learning Pipeline to classify new messages on future disaster events so that the messages can be sent to the appropriate disaster relief agency.

This project includes a web app where an emergency worker can input a new message and get classification results in several categories.


## File Descriptions <a name="files"></a>

Below are additional details about the project structure:

* [/app](https://github.com/marinavillaschi/disaster-response-pipeline/tree/main/app) : contains the Flask webapp files. 

* [/data](https://github.com/marinavillaschi/disaster-response-pipeline/tree/main/data) : contains both .csv files used on the ETL pipeline as well as the `process_data.py` script that holds all the ETL pipeline and the .db result from the ETL pipeline.

* [/models](https://github.com/marinavillaschi/disaster-response-pipeline/tree/main/models) : contains the `train_classifier.py` script that holds the ML pipeline as well as the model pickle file.

* [/notebooks](https://github.com/marinavillaschi/disaster-response-pipeline/tree/main/notebooks)  contains Jupyter Notebooks that were used to build both pipeline scripts.



## Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database

        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disastermanegement.db`

    - To run ML pipeline that trains classifier and saves

        `python models/train_classifier.py data/disastermanagement.db models/message_lr_classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run the web app: `python run.py`

4. Go to http://localhost:3003/


## Licensing, Authors, and Acknowledgements <a name="licensing"></a>

### Licensing

[MIT license](https://github.com/marinavillaschi/disaster-response-pipeline/blob/main/license.txt)

### Authors

[Marina Villaschi](https://www.linkedin.com/in/marinavillaschi/?locale=en_US)

### Acknowledgements:

[Appen](https://appen.com/) (formally Figure 8) for providing the pre-labeled data.

[Udacity](https://www.udacity.com/) and all staff involved for the great guidance and quality course material provided during the Data Science Nanodegree Program.