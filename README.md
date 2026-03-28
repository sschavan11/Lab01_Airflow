# Lab 01 – Airflow-Based Machine Learning Pipeline

## Objective

The objective of this lab is to build and execute a complete machine learning workflow using Apache Airflow.  
The pipeline automates the sequence of loading data, preprocessing features, training a classification model, and evaluating model performance through scheduled tasks.

The workflow uses the Titanic dataset to predict passenger survival and demonstrates how Airflow can manage machine learning tasks in an organized and reproducible manner.

---

## Project Structure

```text
Lab01-Airflow/
│
├── dags/
│   └── titanic_dag.py
│
├── data/
│   └── Titanic-Dataset.csv
│
├── output/
│   ├── titanic_model.pkl
│   └── metrics.txt
│
├── src/
│   ├── __init__.py
│   └── lab.py
│
├── docker-compose.yaml
├── requirements.txt
├── .env
├── README.md

## Workflow Description

The Airflow DAG defines a four-step machine learning pipeline:

- **Load Data**: Reads the Titanic dataset from the local data folder.
- **Preprocess Data**: Selects relevant features, handles missing values, and prepares data for training.
- **Train Model**: Trains a Random Forest Classifier using selected passenger attributes such as age, fare, class, gender, and embarkation point.
- **Evaluate Model**: Generates model accuracy and classification metrics and saves the outputs.

## Airflow DAG Execution

The DAG is created using the TaskFlow API in Apache Airflow.

The pipeline contains the following tasks:

- `load_data_task`
- `preprocess_task`
- `train_task`
- `evaluate_task`

The tasks are executed sequentially inside Airflow and monitored through the Airflow UI.

## Model Used

A Random Forest Classifier is used for classification.

The model is trained after applying:

- Median imputation for numerical columns
- Most frequent value imputation for categorical columns
- One-hot encoding for categorical features

## Output Generated

After successful DAG execution, the following files are generated in the output folder:

- `titanic_model.pkl` → saved trained model
- `metrics.txt` → evaluation metrics

The evaluation file contains:

- Accuracy score
- Precision
- Recall
- F1-score

Example model accuracy achieved:

```
Accuracy: 0.8045
```

## Running the Project

Start Airflow services:

```bash
docker compose up
```

Open Airflow UI:

[http://localhost:8080](http://localhost:8080)

Login credentials:

- username: `airflow`
- password: `airflow`

Trigger the DAG manually from the Airflow interface.

## Result

The Airflow pipeline executed successfully with all tasks completed in sequence, and the machine learning workflow produced both trained model output and evaluation metrics.
