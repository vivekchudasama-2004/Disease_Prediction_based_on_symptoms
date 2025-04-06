# import streamlit as st
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import zipfile
# import shutil
# import numpy as np
# import time
#
# # Page configuration
# st.set_page_config(page_title="Disease-Symptom Dataset Visualization", layout="wide")
# st.title("Dataset Visualization")
#
#
# # Function to extract ZIP file if provided (for new uploads)
# def extract_zip(uploaded_file, temp_folder) :
#     with zipfile.ZipFile(uploaded_file, 'r') as zip_ref :
#         zip_ref.extractall(temp_folder)
#     for file_name in os.listdir(temp_folder) :
#         if file_name.endswith('.csv') :
#             return os.path.join(temp_folder, file_name)
#     return None
#
#
# # Function to preprocess the dataset and create a summary
# def preprocess_data(df) :
#     symptom_cols = [col for col in df.columns if col.startswith("Symptom_")]
#     df = df.dropna(subset=symptom_cols, how='all')  # Drop rows with no symptoms
#
#     # Replace '0' with NaN for proper symptom collection
#     df[symptom_cols] = df[symptom_cols].replace('0', np.nan)
#
#     # Collect all unique symptoms
#     all_symptoms = set()
#     for col in symptom_cols :
#         all_symptoms.update(df[col].dropna().str.strip().str.replace(" ", "_").unique())
#     all_symptoms = sorted(list(all_symptoms))  # Sort for consistency
#
#     # Create binary feature matrix
#     X = pd.DataFrame(0, index=df.index, columns=all_symptoms)
#     for col in symptom_cols :
#         for idx, symptom in df[col].items() :
#             if pd.notna(symptom) :
#                 symptom = symptom.strip().replace(" ", "_")
#                 X.loc[idx, symptom] = 1
#
#     y = df['Disease'].str.strip()
#
#     # Create a summary table: Disease vs Symptom Count
#     summary_data = []
#     diseases = y.unique()
#     for disease in diseases :
#         disease_indices = y[y == disease].index
#         symptom_count = X.loc[disease_indices].sum().sum()  # Total symptoms for this disease
#         num_samples = len(disease_indices)  # Number of samples for this disease
#         summary_data.append([disease, num_samples, int(symptom_count)])
#
#     summary_df = pd.DataFrame(summary_data, columns=['Disease', 'Sample Count', 'Total Symptoms'])
#     summary_df.set_index('Disease', inplace=True)
#
#     return X, y, all_symptoms, summary_df
#
#
# # Function to display disease distribution bar chart
# def display_disease_distribution(y) :
#     st.subheader("Disease Distribution")
#     plt.figure(figsize=(10, 6))
#     disease_counts = y.value_counts()
#     plt.bar(disease_counts.index, disease_counts.values, color='skyblue')
#     plt.xlabel("Disease")
#     plt.ylabel("Count")
#     plt.title("Number of Occurrences per Disease")
#     plt.xticks(rotation=45, ha='right')
#     st.pyplot(plt)
#
#
# # Function to display symptom frequency bar chart
# def display_symptom_frequency(X) :
#     st.subheader("Symptom Frequency Across All Diseases")
#     plt.figure(figsize=(12, 8))
#     symptom_counts = X.sum().sort_values(ascending=False)
#     plt.bar(symptom_counts.index, symptom_counts.values, color='lightgreen')
#     plt.xlabel("Symptom")
#     plt.ylabel("Frequency")
#     plt.title("Frequency of Each Symptom")
#     plt.xticks(rotation=45, ha='right')
#     st.pyplot(plt)
#
#
# # Function to display pie chart for all diseases
# def display_disease_pie_chart(y) :
#     st.subheader("Distribution of All Diseases")
#     disease_counts = y.value_counts()
#     labels = disease_counts.index
#     sizes = disease_counts.values
#     plt.figure(figsize=(16, 16))
#     plt.pie(
#         sizes,
#         labels=labels,
#         autopct='%1.1f%%',
#         startangle=90,
#         colors=plt.cm.tab20c(range(len(labels)))
#     )
#     plt.title("Distribution of All Diseases")
#     st.pyplot(plt)
#
#
# # Function to display symptom co-occurrence heatmap
# def display_symptom_cooccurrence(X) :
#     st.subheader("Symptom Co-occurrence Heatmap")
#     cooccurrence_matrix = X.T.dot(X)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cooccurrence_matrix, cmap="YlGnBu", annot=False)
#     plt.title("Symptom Co-occurrence Heatmap")
#     plt.xlabel("Symptoms")
#     plt.ylabel("Symptoms")
#     st.pyplot(plt)
#
#
# # Function to display disease-symptom frequency heatmap
# def display_disease_symptom_heatmap(X, y) :
#     st.subheader("Disease-Symptom Frequency Heatmap")
#     disease_symptom_matrix = pd.DataFrame(0, index=y.unique(), columns=X.columns)
#     for disease in y.unique() :
#         disease_indices = y[y == disease].index
#         disease_symptom_matrix.loc[disease] = X.loc[disease_indices].sum()
#
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(disease_symptom_matrix, cmap="Blues", annot=False)
#     plt.title("Disease-Symptom Frequency Heatmap")
#     plt.xlabel("Symptoms")
#     plt.ylabel("Diseases")
#     plt.xticks(rotation=45, ha='right')
#     st.pyplot(plt)
#
#
# # Main Visualization Logic
# desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
# dataset_folder = os.path.join(desktop_path, "Dataset")
#
# if not os.path.exists(dataset_folder) :
#     st.warning("No datasets found at ~/Desktop/Dataset. Please upload a dataset using uploaddata.py.")
#     st.stop()
#
# dataset_types = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
# if not dataset_types :
#     dataset_types = ["text_classification"]  # Default from uploaddata.py
#     os.makedirs(os.path.join(dataset_folder, "text_classification"))
#
# dataset_type = st.selectbox("Select Dataset Type", dataset_types)
# dataset_type_folder = os.path.join(dataset_folder, dataset_type)
#
# models = [f for f in os.listdir(dataset_type_folder) if os.path.isdir(os.path.join(dataset_type_folder, f))]
# if not models :
#     st.warning("No models found in the selected dataset type. Please upload a dataset using uploaddata.py.")
#     st.stop()
#
# st.subheader("Model Selection")
# add_new_model = st.checkbox("Add New Model for Visualization")
# if add_new_model :
#     new_model_name = st.text_input("Enter New Model Name for Visualization")
#     if new_model_name and new_model_name not in models :
#         uploaded_file = st.file_uploader("Upload dataset.csv or a ZIP containing dataset.csv", type=["csv", "zip"])
#     else :
#         uploaded_file = None
#         if new_model_name in models :
#             st.warning(
#                 f"Model '{new_model_name}' already exists. Please choose a different name or select an existing model.")
# else :
#     selected_model = st.selectbox("Select Existing Model", models)
#     model_folder = os.path.join(dataset_type_folder, selected_model)
#     versions = [f for f in os.listdir(model_folder) if os.path.isdir(os.path.join(model_folder, f))]
#     if not versions :
#         st.warning(f"No versions found for model '{selected_model}'. Please upload a dataset using uploaddata.py.")
#         st.stop()
#     selected_version = st.selectbox("Select Version", versions)
#     uploaded_file = None
#
# # Load dataset based on selection
# if add_new_model and uploaded_file :
#     temp_folder = os.path.join(os.path.expanduser("~"), "temp_dataset_viz")
#     os.makedirs(temp_folder, exist_ok=True)
#
#     if uploaded_file.name.endswith('.zip') :
#         csv_path = extract_zip(uploaded_file, temp_folder)
#         if csv_path :
#             df = pd.read_csv(csv_path)
#         else :
#             st.error("No CSV file found in the ZIP!")
#             st.stop()
#     else :
#         df = pd.read_csv(uploaded_file)
# elif not add_new_model and selected_model and selected_version :
#     train_path = os.path.join(dataset_type_folder, selected_model, selected_version, "train", "train.csv")
#     if os.path.exists(train_path) :
#         df = pd.read_csv(train_path)
#     else :
#         st.error(
#             f"Training dataset not found at {train_path}. Please upload a dataset using uploaddata.py with the correct type, model, and version.")
#         st.stop()
# else :
#     st.write("Please select an existing model and version or add a new model and upload a dataset.")
#     st.stop()
#
# # Preprocess and visualize
# X, y, all_symptoms, summary_df = preprocess_data(df)
#
# st.subheader("Dataset Summary Table")
# st.dataframe(summary_df.style.set_properties(**{'text-align' : 'center'}))
#
# if st.button("Show Dataset Visualization") :
#     col1, col2 = st.columns(2)
#
#     with col1 :
#         display_disease_distribution(y)
#         display_disease_pie_chart(y)
#
#     with col2 :
#         display_symptom_frequency(X)
#         display_symptom_cooccurrence(X)
#
#     # Display the disease-symptom heatmap
#     display_disease_symptom_heatmap(X, y)
#
#     # Clean up temporary folder if used
#     if add_new_model and uploaded_file and os.path.exists(temp_folder) :
#         try :
#             time.sleep(1)
#             shutil.rmtree(temp_folder)
#             st.success(f"Temporary folder {temp_folder} deleted successfully.")
#         except Exception as e :
#             st.warning(f"Failed to delete temporary folder {temp_folder}: {e}")

import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import zipfile
import shutil
import numpy as np
import time

# Page configuration
st.set_page_config(page_title="Disease-Symptom Dataset Visualization", layout="wide")
st.title("Dataset Visualization")


# Function to extract ZIP file if provided (for new uploads)
def extract_zip(uploaded_file, temp_folder) :
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref :
        zip_ref.extractall(temp_folder)
    for file_name in os.listdir(temp_folder) :
        if file_name.endswith('.csv') :
            return os.path.join(temp_folder, file_name)
    return None


# Function to preprocess the dataset and create a summary
def preprocess_data(df) :
    symptom_cols = [col for col in df.columns if col.startswith("Symptom_")]
    df = df.dropna(subset=symptom_cols, how='all')  # Drop rows with no symptoms

    # Replace '0' with NaN for proper symptom collection
    df[symptom_cols] = df[symptom_cols].replace('0', np.nan)

    # Collect all unique symptoms
    all_symptoms = set()
    for col in symptom_cols :
        all_symptoms.update(df[col].dropna().str.strip().str.replace(" ", "_").unique())
    all_symptoms = sorted(list(all_symptoms))  # Sort for consistency

    # Create binary feature matrix
    X = pd.DataFrame(0, index=df.index, columns=all_symptoms)
    for col in symptom_cols :
        for idx, symptom in df[col].items() :
            if pd.notna(symptom) :
                symptom = symptom.strip().replace(" ", "_")
                X.loc[idx, symptom] = 1

    y = df['Disease'].str.strip()

    # Create a summary table: Disease vs Symptom Count
    summary_data = []
    diseases = y.unique()
    for disease in diseases :
        disease_indices = y[y == disease].index
        symptom_count = X.loc[disease_indices].sum().sum()  # Total symptoms for this disease
        num_samples = len(disease_indices)  # Number of samples for this disease
        summary_data.append([disease, num_samples, int(symptom_count)])

    summary_df = pd.DataFrame(summary_data, columns=['Disease', 'Sample Count', 'Total Symptoms'])
    summary_df.set_index('Disease', inplace=True)

    return X, y, all_symptoms, summary_df


# Function to display disease distribution bar chart (animated with Plotly)
def display_disease_distribution(y) :
    st.subheader("Disease Distribution")
    disease_counts = y.value_counts().reset_index()
    disease_counts.columns = ['Disease', 'Count']

    fig = px.bar(
        disease_counts,
        x='Disease',
        y='Count',
        color='Count',
        title="Number of Occurrences per Disease",
        height=500,
        text_auto=True,
        animation_frame=None  # Static but hover-interactive
    )
    fig.update_traces(hovertemplate='Disease: %{x}<br>Count: %{y}')
    fig.update_layout(xaxis={'tickangle' : -45}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# Function to display symptom frequency bar chart (animated with Plotly)
def display_symptom_frequency(X) :
    st.subheader("Symptom Frequency Across All Diseases")
    symptom_counts = X.sum().sort_values(ascending=False).reset_index()
    symptom_counts.columns = ['Symptom', 'Frequency']

    fig = px.bar(
        symptom_counts,
        x='Symptom',
        y='Frequency',
        color='Frequency',
        title="Frequency of Each Symptom",
        height=500,
        text_auto=True
    )
    fig.update_traces(hovertemplate='Symptom: %{x}<br>Frequency: %{y}')
    fig.update_layout(xaxis={'tickangle' : -45}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# Function to display pie chart for all diseases (animated with Plotly)
def display_disease_pie_chart(y) :
    st.subheader("Distribution of All Diseases")
    disease_counts = y.value_counts().reset_index()
    disease_counts.columns = ['Disease', 'Count']

    fig = px.pie(
        disease_counts,
        names='Disease',
        values='Count',
        title="Distribution of All Diseases",
        height=600,
        hole=0.3  # Donut style for better aesthetics
    )
    fig.update_traces(textinfo='percent+label',
                      hovertemplate='Disease: %{label}<br>Count: %{value}<br>Percentage: %{percent}')
    st.plotly_chart(fig, use_container_width=True)


# Function to display symptom co-occurrence heatmap (animated with Plotly)
def display_symptom_cooccurrence(X) :
    st.subheader("Symptom Co-occurrence Heatmap")
    cooccurrence_matrix = X.T.dot(X)

    fig = px.imshow(
        cooccurrence_matrix,
        labels=dict(x="Symptoms", y="Symptoms", color="Co-occurrence"),
        title="Symptom Co-occurrence Heatmap",
        height=600
    )
    fig.update_traces(hovertemplate='Symptom X: %{x}<br>Symptom Y: %{y}<br>Co-occurrence: %{z}')
    st.plotly_chart(fig, use_container_width=True)


# Function to display disease-symptom frequency heatmap (animated with Plotly)
def display_disease_symptom_heatmap(X, y) :
    st.subheader("Disease-Symptom Frequency Heatmap")
    disease_symptom_matrix = pd.DataFrame(0, index=y.unique(), columns=X.columns)
    for disease in y.unique() :
        disease_indices = y[y == disease].index
        disease_symptom_matrix.loc[disease] = X.loc[disease_indices].sum()

    fig = px.imshow(
        disease_symptom_matrix,
        labels=dict(x="Symptoms", y="Diseases", color="Frequency"),
        title="Disease-Symptom Frequency Heatmap",
        height=600
    )
    fig.update_traces(hovertemplate='Disease: %{y}<br>Symptom: %{x}<br>Frequency: %{z}')
    fig.update_layout(xaxis={'tickangle' : -45})
    st.plotly_chart(fig, use_container_width=True)


# Main Visualization Logic
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
dataset_folder = os.path.join(desktop_path, "Dataset")

if not os.path.exists(dataset_folder) :
    st.warning("No datasets found at ~/Desktop/Dataset. Please upload a dataset using uploaddata.py.")
    st.stop()

dataset_types = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
if not dataset_types :
    dataset_types = ["text_classification"]  # Default from uploaddata.py
    os.makedirs(os.path.join(dataset_folder, "text_classification"))

dataset_type = st.selectbox("Select Dataset Type", dataset_types)
dataset_type_folder = os.path.join(dataset_folder, dataset_type)

models = [f for f in os.listdir(dataset_type_folder) if os.path.isdir(os.path.join(dataset_type_folder, f))]
if not models :
    st.warning("No models found in the selected dataset type. Please upload a dataset using uploaddata.py.")
    st.stop()

st.subheader("Model Selection")
add_new_model = st.checkbox("Add New Model for Visualization")
if add_new_model :
    new_model_name = st.text_input("Enter New Model Name for Visualization")
    if new_model_name and new_model_name not in models :
        uploaded_file = st.file_uploader("Upload dataset.csv or a ZIP containing dataset.csv", type=["csv", "zip"])
    else :
        uploaded_file = None
        if new_model_name in models :
            st.warning(
                f"Model '{new_model_name}' already exists. Please choose a different name or select an existing model.")
else :
    selected_model = st.selectbox("Select Existing Model", models)
    model_folder = os.path.join(dataset_type_folder, selected_model)
    versions = [f for f in os.listdir(model_folder) if os.path.isdir(os.path.join(model_folder, f))]
    if not versions :
        st.warning(f"No versions found for model '{selected_model}'. Please upload a dataset using uploaddata.py.")
        st.stop()
    selected_version = st.selectbox("Select Version", versions)
    uploaded_file = None

# Load dataset based on selection
if add_new_model and uploaded_file :
    temp_folder = os.path.join(os.path.expanduser("~"), "temp_dataset_viz")
    os.makedirs(temp_folder, exist_ok=True)

    if uploaded_file.name.endswith('.zip') :
        csv_path = extract_zip(uploaded_file, temp_folder)
        if csv_path :
            df = pd.read_csv(csv_path)
        else :
            st.error("No CSV file found in the ZIP!")
            st.stop()
    else :
        df = pd.read_csv(uploaded_file)
elif not add_new_model and selected_model and selected_version :
    train_path = os.path.join(dataset_type_folder, selected_model, selected_version, "train", "train.csv")
    if os.path.exists(train_path) :
        df = pd.read_csv(train_path)
    else :
        st.error(
            f"Training dataset not found at {train_path}. Please upload a dataset using uploaddata.py with the correct type, model, and version.")
        st.stop()
else :
    st.write("Please select an existing model and version or add a new model and upload a dataset.")
    st.stop()

# Preprocess and visualize
X, y, all_symptoms, summary_df = preprocess_data(df)

st.subheader("Dataset Summary Table")
st.dataframe(summary_df.style.set_properties(**{'text-align' : 'center'}))

if st.button("Show Dataset Visualization") :
    col1, col2 = st.columns(2)

    with col1 :
        display_disease_distribution(y)
        display_disease_pie_chart(y)

    with col2 :
        display_symptom_frequency(X)
        display_symptom_cooccurrence(X)

    # Display the disease-symptom heatmap
    display_disease_symptom_heatmap(X, y)

    # Clean up temporary folder if used
    if add_new_model and uploaded_file and os.path.exists(temp_folder) :
        try :
            time.sleep(1)
            shutil.rmtree(temp_folder)
            st.success(f"Temporary folder {temp_folder} deleted successfully.")
        except Exception as e :
            st.warning(f"Failed to delete temporary folder {temp_folder}: {e}")