import streamlit as st
import os
import shutil
from datetime import datetime
import mlflow
from huggingface_hub import HfApi, Repository, create_repo, upload_file, login
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Hugging Face credentials
hf_username = "Raj23804"
hf_token = "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"
login(token=hf_token)  # Authenticate once at startup

# Predefined list of classification models
CLASSIFICATION_MODELS = {
    "XGBoost" : "xgboost",
    "Random Forest" : "random_forest",
    "Decision Tree" : "decision_tree",
}


# Functions for Hugging Face and MLflow (unchanged)
def create_huggingface_repo(model_name) :
    try :
        dynamic_repo_name = model_name.replace(" ", "_")
        create_repo(repo_id=f"{hf_username}/{dynamic_repo_name}", token=hf_token, exist_ok=True, private=False)
        return dynamic_repo_name
    except Exception as e :
        st.error(f"Failed to create Hugging Face repo: {e}")
        return None


def upload_to_huggingface(model_path, repo_name) :
    if os.path.exists(model_path) :
        try :
            upload_file(
                path_or_fileobj=model_path,
                path_in_repo=os.path.basename(model_path),
                repo_id=f"{hf_username}/{repo_name}",
                token=hf_token,
            )
            st.success(f"Model successfully uploaded to Hugging Face repository: {repo_name}")
        except Exception as e :
            st.error(f"Failed to upload model: {e}")
    else :
        st.error(f"Model file not found: {model_path}")


def get_unique_temp_folder(base_folder) :
    counter = 1
    temp_folder = "D:/temp"
    while os.path.exists(f"{temp_folder}{counter}") :
        counter += 1
    return f"{temp_folder}{counter}"


def generate_model_run_name(selected_model_name, dataset_type) :
    model_prefix = selected_model_name.lower().replace(" ", "_")
    current_date = datetime.now().strftime("%Y%m%d")
    base_model_run_name = f"{model_prefix}_{dataset_type}_{current_date}"
    counter = 0
    while True :
        versioned_run_name = f"{base_model_run_name}_v{counter}"
        experiment_name = f"{selected_model_name}"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None :
            experiment_id = mlflow.create_experiment(experiment_name)
        else :
            experiment_id = experiment.experiment_id
        run = mlflow.search_runs(experiment_ids=experiment_id,
                                 filter_string=f"tags.model_run_name = '{versioned_run_name}'")
        if run.empty :
            break
        counter += 1
    return versioned_run_name, experiment_id


# Load and preprocess data from uploaded dataset
def load_and_preprocess_data(train_path, valid_path, test_path) :
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    # Assuming symptom_weights.csv is available
    symptoms_weights = pd.read_csv("symptom_weights.csv")  # Adjust path as needed
    all_symptoms = symptoms_weights["Symptom"].tolist()

    def prepare_features(df) :
        X = pd.DataFrame(0, index=range(len(df)), columns=all_symptoms)
        for idx, row in df.iterrows() :
            for symptom_col in [f"Symptom_{i}" for i in range(1, 18)] :
                symptom = row.get(symptom_col)
                if pd.notna(symptom) and symptom in all_symptoms :
                    X.loc[idx, symptom] = 1  # Or use weights if desired
        return X

    X_train = prepare_features(train_df)
    X_valid = prepare_features(valid_df)
    X_test = prepare_features(test_df)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["Disease"])
    y_valid = le.transform(valid_df["Disease"])
    y_test = le.transform(test_df["Disease"])

    return X_train, X_valid, X_test, y_train, y_valid, y_test, le


# Streamlit UI
st.title("Tabular Disease Classification Model Training")

# Dataset selection from uploaded data
st.subheader("Select Dataset")
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
dataset_folder = os.path.join(desktop_path, "Dataset")

if os.path.exists(dataset_folder) :
    dataset_types = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
    if not dataset_types :
        st.warning("No dataset types found at ~/Desktop/Dataset. Please upload a dataset using uploaddata.py.")
        st.stop()
else :
    st.warning("Dataset folder not found at ~/Desktop/Dataset. Please upload a dataset using uploaddata.py.")
    os.makedirs(dataset_folder)
    st.stop()

dataset_type = st.selectbox("Select Dataset Type", dataset_types)
dataset_type_folder = os.path.join(dataset_folder, dataset_type)

if os.path.exists(dataset_type_folder) :
    models = [f for f in os.listdir(dataset_type_folder) if os.path.isdir(os.path.join(dataset_type_folder, f))]
else :
    models = []
    os.makedirs(dataset_type_folder)

if not models :
    st.warning("No models found in the selected dataset type. Please upload a dataset using uploaddata.py.")
    st.stop()

selected_model = st.selectbox("Select Existing Model", models)
model_folder = os.path.join(dataset_type_folder, selected_model)

versions = [f for f in os.listdir(model_folder) if os.path.isdir(os.path.join(model_folder, f))]
if not versions :
    st.warning(f"No versions found for model '{selected_model}'. Please upload a dataset using uploaddata.py.")
    st.stop()

selected_version = st.selectbox("Select Dataset Version", versions)
version_folder = os.path.join(model_folder, selected_version)

# Verify dataset files exist
train_path = os.path.join(version_folder, "train", "train.csv")
valid_path = os.path.join(version_folder, "valid", "valid.csv")
test_path = os.path.join(version_folder, "test", "test.csv")

if not all(os.path.exists(p) for p in [train_path, valid_path, test_path]) :
    st.error(f"Dataset files missing in {version_folder}. Please upload a complete dataset using uploaddata.py.")
    st.stop()

st.write(f"Using dataset: {selected_model}/{selected_version}")

# Model selection
st.subheader("Select Model")
selected_model_name = st.selectbox("Select Classification Model", list(CLASSIFICATION_MODELS.keys()))

# Training Parameters
st.subheader("Training Parameters")
initial_model_run_name, experiment_id = generate_model_run_name(selected_model_name, dataset_type)
model_run_name = st.text_input("Model Run Name (Editable)", initial_model_run_name)
trainer_name = st.text_input("Trainer Name", "Trainer Name")
description = st.text_area("Enter Model Description", "")
n_estimators = st.number_input("Number of Estimators", min_value=10, value=100)
max_depth = st.number_input("Max Depth", min_value=1, value=6)
learning_rate = st.number_input("Learning Rate", min_value=0.00001, max_value=1.0, value=0.1, format="%.5f")

# Check if a run is already in progress
if 'is_training' not in st.session_state :
    st.session_state.is_training = False

# Start Training Button
if st.button("Start Training") :
    if st.session_state.is_training :
        st.warning("A training process is already in progress. Please wait.")
    else :
        st.session_state.is_training = True

        temp_folder = get_unique_temp_folder(model_run_name)
        os.makedirs(temp_folder, exist_ok=True)

        # Load and preprocess data
        X_train, X_valid, X_test, y_train, y_valid, y_test, label_encoder = load_and_preprocess_data(
            train_path, valid_path, test_path
        )

        # Set up MLflow tracking
        mlflow.set_tracking_uri("http://localhost:5000")
        with mlflow.start_run(experiment_id=experiment_id, run_name=model_run_name) :
            mlflow.log_param("Model Name", selected_model_name)
            mlflow.log_param("Trainer Name", trainer_name)
            mlflow.log_param("Number of Estimators", n_estimators)
            mlflow.log_param("Max Depth", max_depth)
            mlflow.log_param("Learning Rate", learning_rate)
            mlflow.set_tag("model_run_name", model_run_name)
            run_id = mlflow.active_run().info.run_id

            try :
                # Train the model based on selection
                if selected_model_name == "XGBoost" :
                    model = XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        use_label_encoder=False,
                        eval_metric="mlogloss"
                    )
                elif selected_model_name == "Random Forest" :
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth
                    )

                model.fit(X_train, y_train)

                # Evaluate model
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                mlflow.log_metric("train_accuracy", train_score)
                mlflow.log_metric("test_accuracy", test_score)

                # Save the model
                model_path = os.path.join(temp_folder, f"{model_run_name}.pkl")
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path)

                final_path = f"D:/Project_Raj/artifacts/{experiment_id}/{run_id}/artifacts/{model_run_name}.pkl"
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                shutil.copy(model_path, final_path)

                st.info(f"Model saved to {final_path}")

                # Upload to Hugging Face
                repo_name = create_huggingface_repo(model_run_name)
                if repo_name :
                    upload_to_huggingface(model_path, repo_name)

                st.success(f"Training completed successfully for {model_run_name}!")

            except Exception as e :
                st.error(f"Training failed: {e}")

        mlflow_url = "http://localhost:5000"
        st.markdown(f"Training is complete! View logs in the [MLflow Dashboard]({mlflow_url}).")
        st.session_state.is_training = False