import streamlit as st
import os
import pandas as pd
import random
import shutil
import tempfile

# Title for the dataset upload interface
st.title("Upload Dataset")

# Tabs for different modes of uploading
tab1, tab2 = st.tabs(["Upload Dataset for Model Training", "Upload Dataset for Benchmark Model"])


def create_folders(version_folder) :
    # Create train, valid, and test directories
    train_dir = os.path.join(version_folder, 'train')
    valid_dir = os.path.join(version_folder, 'valid')
    test_dir = os.path.join(version_folder, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    return train_dir, valid_dir, test_dir


def split_and_save_csv(csv_file_path, version_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1) :
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1."

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Compute split sizes
    total_rows = len(df)
    train_size = int(total_rows * train_ratio)
    val_size = int(total_rows * val_ratio)

    # Split the DataFrame
    train_df = df[:train_size]
    val_df = df[train_size :train_size + val_size]
    test_df = df[train_size + val_size :]

    # Create directories
    train_dir, valid_dir, test_dir = create_folders(version_folder)

    # Save the split DataFrames as CSV files
    train_df.to_csv(os.path.join(train_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(valid_dir, 'valid.csv'), index=False)
    test_df.to_csv(os.path.join(test_dir, 'test.csv'), index=False)

    st.success(f"Dataset successfully split and saved into {version_folder} as train.csv, valid.csv, and test.csv.")


# Streamlit User Interface
with tab1 :
    # Get the path to the Desktop Dataset directory
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    dataset_folder = os.path.join(desktop_path, "Dataset")

    # Fetch dataset types from the Dataset directory
    if os.path.exists(dataset_folder) :
        dataset_types = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
        if not dataset_types :
            dataset_types = ["text_classification"]  # Default for text-based data
    else :
        dataset_types = ["text_classification"]

    # Select Dataset Type
    dataset_type = st.selectbox("Select Dataset Type", dataset_types, key="tab1_selectbox")

    # Create dataset folder if it doesn't exist
    if not os.path.exists(dataset_folder) :
        os.makedirs(dataset_folder)

    # Create dataset type folder if it doesn't exist
    dataset_type_folder = os.path.join(dataset_folder, dataset_type)
    if not os.path.exists(dataset_type_folder) :
        os.makedirs(dataset_type_folder)

    # Get the list of models in the dataset type folder
    models = [f for f in os.listdir(dataset_type_folder) if os.path.isdir(os.path.join(dataset_type_folder, f))]

    # Add New Model Option
    add_new_model = st.checkbox("Add New Model", key="tab1_add_new_model")

    if add_new_model :
        new_model_name = st.text_input("Enter New Model Name", key="tab1_new_model_name")
        if new_model_name :
            if new_model_name not in models :
                models.append(new_model_name)
                new_model_dir = os.path.join(dataset_type_folder, new_model_name)
                os.makedirs(new_model_dir, exist_ok=True)
                st.success(f"Model '{new_model_name}' added successfully!")
            else :
                st.warning(f"Model '{new_model_name}' already exists.")
        existing_model = new_model_name
    else :
        if models :
            existing_model = st.selectbox("Select Existing Model", models, key="tab1_existing_model")
        else :
            st.warning("No existing models found. Please add a new model.")
            existing_model = None

    # Dataset version name
    dataset_version = st.text_input("Dataset Version Name", placeholder="Enter version name (e.g., version_1)",
                                    key="tab1_version")

    # File uploader for CSV
    uploaded_file = st.file_uploader(
        "Upload a CSV file", type=["csv"], accept_multiple_files=False,
        help="Upload a CSV file with disease and symptom data", key="tab1_file_uploader"
    )

    # Optional description
    description = st.text_area("Description", placeholder="Optional: Add a description for this dataset version",
                               key="tab1_description")

    # Checkbox for allowing user to change splitting ratios
    change_split_ratio = st.checkbox("Change Splitting Ratios", key="tab1_change_split_ratio")

    # User-defined ratios for splitting
    if change_split_ratio :
        st.subheader("Splitting Ratio")
        train_ratio = st.number_input("Training Ratio", min_value=0.0, max_value=1.0, value=0.8, step=0.01,
                                      key="tab1_train_ratio")
        val_ratio = st.number_input("Validation Ratio", min_value=0.0, max_value=1.0, value=0.1, step=0.01,
                                    key="tab1_val_ratio")
        test_ratio = st.number_input("Testing Ratio", min_value=0.0, max_value=1.0, value=0.1, step=0.01,
                                     key="tab1_test_ratio")
    else :
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1

    # Upload and process the CSV file
    if st.button("Upload and Split", key="tab1_upload_button") :
        if uploaded_file is not None and existing_model and dataset_version :
            if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5 :
                st.error("The sum of the ratios must be equal to 1.")
            else :
                # Create model folder
                model_folder = os.path.join(dataset_type_folder, existing_model)
                if not os.path.exists(model_folder) :
                    os.makedirs(model_folder)

                # Create version folder
                version_folder = os.path.join(model_folder, dataset_version)
                if not os.path.exists(version_folder) :
                    os.makedirs(version_folder)

                # Save the uploaded CSV temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file :
                    temp_file.write(uploaded_file.getbuffer())
                    temp_csv_path = temp_file.name

                # Split and save the CSV data
                split_and_save_csv(temp_csv_path, version_folder, train_ratio, val_ratio, test_ratio)

                # Clean up temporary file
                os.remove(temp_csv_path)
        else :
            st.error("Please upload a CSV file, select or add a model, and provide a version name.")

with tab2 :
    # Benchmark tab (simplified for now, can be expanded later)
    st.write("Upload a CSV file for benchmarking a model.")
    uploaded_benchmark_file = st.file_uploader(
        "Upload CSV File", type=["csv"], accept_multiple_files=False, key="tab2_file_uploader"
    )
    if uploaded_benchmark_file :
        # Save the file to a specific location (e.g., D:/Benchmark)
        benchmark_path = "D:/Benchmark"
        if not os.path.exists(benchmark_path) :
            os.makedirs(benchmark_path)
        benchmark_file_path = os.path.join(benchmark_path, "benchmark.csv")
        with open(benchmark_file_path, "wb") as f :
            f.write(uploaded_benchmark_file.getbuffer())
        st.success(f"Benchmark CSV file saved to: {benchmark_file_path}")

if __name__ == "__main__" :
    st.write("Ready to process your dataset!")