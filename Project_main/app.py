import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit_option_menu as som
import os

# File paths
BASE_PATH = r'C:\Users\Vivek\PycharmProjects\Project\Project_main\Dataset'
MODEL_PATH = 'model_RFC.sav'
SEVERITY_PATH = os.path.join(BASE_PATH, 'Symptom-severity.csv')
PRECAUTION_PATH = os.path.join(BASE_PATH, 'symptom_precaution.csv')
DESCRIPTION_PATH = os.path.join(BASE_PATH, 'symptom_description.csv')
DATASET_PATH = os.path.join(BASE_PATH, 'dataset.csv')

# Load the model
try :
    model_RFC = jb.load(MODEL_PATH)
except FileNotFoundError :
    st.error(f"Model file not found at {MODEL_PATH}. Please ensure it exists.")
    st.stop()


# Load additional datasets with error handling
def load_csv(path, name) :
    if os.path.exists(path) :
        return pd.read_csv(path)
    else :
        st.error(f"{name} file not found at {path}. Please check the path.")
        st.stop()


severity = load_csv(SEVERITY_PATH, 'Symptom-severity')
precautions = load_csv(PRECAUTION_PATH, 'Symptom-precaution')
descriptions = load_csv(DESCRIPTION_PATH, 'description')

# Clean the datasets
severity['Symptom'] = severity['Symptom'].str.replace('_', ' ').str.strip()
precautions['Disease'] = precautions['Disease'].str.strip()
descriptions['Disease'] = descriptions['Disease'].str.strip()

# Custom CSS with improved color scheme
st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #1a3c34; /* Deep teal for a professional look */
        text-align: center;
        margin-bottom: 20px;
    }
    .prediction-box {
        background-color: #e6f3f8; /* Soft light blue */
        padding: 20px;
        color:black;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); /* Softer shadow */
        margin-bottom: 20px;
        border-left: 5px solid #3498db; /* Blue accent */
    }
    .severity-item {
        background-color: #fef5e7; /* Light peach for warmth */
        color: #7f4c00; /* Darker text for contrast */
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #f39c12; /* Orange accent */
    }
    .precaution-item {
        background-color: #e9f7ef; /* Light mint green */
        color: #1a3c34; /* Dark teal text */
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #27ae60; /* Green accent */
    }
    .description-box {
        background-color: #f8f9fa; /* Light gray */
        color: #2c3e50; /* Dark slate for readability */
        padding: 15px;
        border-radius: 5px;
        font-style: italic;
        border-left: 5px solid #95a5a6; /* Subtle gray accent */
    }
    .stButton>button {
        background-color: #3498db; /* Blue button */
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #2980b9; /* Darker blue on hover */
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar menu
with st.sidebar :
    menu_option = ['Prediction', 'Add Data', 'Train Model']
    selected_option = som.option_menu('Disease Prediction System Based on Symptoms', options=menu_option,
                                      icons=['hospital', 'database-fill-add', 'train-front'], menu_icon='bandaid')

# Prediction page
if selected_option == 'Prediction' :
    st.title('Disease Prediction System')
    st.write("Select symptoms below to predict the disease and get detailed insights.")

    symptom_list = severity['Symptom'].tolist()
    selected_symptoms = st.multiselect('Select Symptoms', options=symptom_list, help="Choose up to 17 symptoms.")


    def prediction(selected_symptoms) :
        data = [0] * 17
        severity_dict = dict(zip(severity['Symptom'], severity['weight']))

        for i, symptom in enumerate(selected_symptoms[:17]) :
            symptom = symptom.lower().strip()
            data[i] = severity_dict.get(symptom, 0)

        data = np.array(data, dtype=float).reshape(1, -1)
        pred = model_RFC.predict(data)[0]

        severity_info = [(symptom, severity_dict.get(symptom.lower(), 'Unknown')) for symptom in selected_symptoms]
        precaution_info = precautions[precautions['Disease'] == pred].iloc[0, 1 :].dropna().tolist() if pred in \
                                                                                                        precautions[
                                                                                                            'Disease'].values else [
            'No precautions available']
        description_info = descriptions[descriptions['Disease'] == pred]['Description'].values[0] if pred in \
                                                                                                     descriptions[
                                                                                                         'Disease'].values else 'No description available'

        return pred, severity_info, precaution_info, description_info


    if st.button('Make Prediction', key='predict_btn') :
        if selected_symptoms :
            with st.spinner("Predicting...") :
                dia_prediction, severity_info, precaution_info, description_info = prediction(selected_symptoms)

            # Prediction Output in a styled container
            with st.container() :
                st.markdown(f'<div class="prediction-box"><h3>Predicted Disease: {dia_prediction}</h3></div>',
                            unsafe_allow_html=True)
                st.subheader(f"Predicted Disease: {dia_prediction}", anchor=None)
                st.markdown('</div>', unsafe_allow_html=True)

            # Use columns for a cleaner layout
            col1, col2 = st.columns(2)

            with col1 :
                with st.expander("Symptom Severity", expanded=True) :
                    for symptom, weight in severity_info :
                        st.markdown(f'<div class="severity-item"><b>{symptom}</b>: Severity {weight}</div>',
                                    unsafe_allow_html=True)

            with col2 :
                with st.expander("Precautions", expanded=True) :
                    for precaution in precaution_info :
                        st.markdown(f'<div class="precaution-item">{precaution}</div>', unsafe_allow_html=True)

            # Description in a separate styled box
            with st.container() :
                st.subheader("Description")
                st.markdown(f'<div class="description-box">{description_info}</div>', unsafe_allow_html=True)

        else :
            st.error('Please select at least one symptom.')

# Add Data page
elif selected_option == 'Add Data' :
    st.title('Your Contribution is Valuable! 🩺')
    st.write('#### Please provide the necessary data below to enhance the system.')
    st.info('📝 **Instructions:** Select the symptoms and provide the corresponding label (disease name).')

    symptom_list = severity['Symptom'].tolist()
    label = st.text_input('Disease Label (e.g., Flu, COVID-19, etc.):')
    selected_symptoms = st.multiselect('Symptoms:', options=symptom_list)

    if len(selected_symptoms) > 17 :
        st.warning('⚠️ Please select no more than 17 symptoms.')


    def add_data(label, selected_symptoms) :
        data = [label] + ['0'] * 17
        for i, symptom in enumerate(selected_symptoms[:17]) :
            data[i + 1] = symptom

        dataset = load_csv(DATASET_PATH, 'dataset')
        df = pd.DataFrame([data], columns=dataset.columns)
        dataset = pd.concat([dataset, df], ignore_index=True)
        dataset.to_csv(DATASET_PATH, index=False)
        return 'Data added successfully!'


    if st.button("Submit") :
        if label and selected_symptoms :
            result = add_data(label, selected_symptoms)
            st.success('✅ Data insertion complete. Thank you for your contribution! 🤗')
        else :
            st.error('❌ Please provide a label and select at least one symptom.')

# Train Model page
elif selected_option == 'Train Model' :
    st.title('Model Training Page')
    st.header("Train the Model")
    st.write("Click the button to start training the model.")


    def training_model() :
        dataset = load_csv(DATASET_PATH, 'dataset')
        for col in dataset.columns :
            dataset[col] = dataset[col].str.replace('_', ' ', regex=False).str.strip()
        dataset.fillna('0', inplace=True)

        severity_dict = dict(zip(severity['Symptom'], severity['weight']))
        vals = dataset.values
        for i in range(1, dataset.shape[1]) :
            for j in range(len(vals)) :
                vals[j, i] = severity_dict.get(vals[j, i], 0)

        dataset = pd.DataFrame(vals, columns=dataset.columns)
        dataset = dataset.replace(['spotting  urination', 'dischromic  patches', 'foul smell of urine'], 0)

        X = dataset.iloc[:, 1 :].values.astype(float)
        y = dataset['Disease'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_RFC.fit(X_train, y_train)
        pred = model_RFC.predict(X_test)
        acc = accuracy_score(y_test, pred)

        jb.dump(model_RFC, MODEL_PATH)
        return acc


    if st.button("Start Training") :
        with st.spinner("Training...") :
            acc = training_model()
        st.success("Model Trained Successfully!")
        st.success(f"The Accuracy of the model is: {acc * 100:.2f}%")
