# Disease Prediction Based on Symptoms

A user-friendly machine learning project that predicts possible diseases based on symptoms provided by the user. Built for students, data scientists, and healthcare enthusiasts to explore AI-assisted early disease detection.

---

## 🚀 Overview

This project leverages machine learning to analyze user symptoms and predict possible diseases. Users can interact via a web application or Jupyter Notebook.  
**Note:** This tool is for educational purposes and not a substitute for professional medical advice.

---

## ✨ Features

- **Easy-to-use Web App:** Powered by `app.py` for seamless symptom input and predictions.
- **Interactive Notebook:** Explore data, training, and predictions within `Disease_Prediction_based_on_symptoms.ipynb`.
- **Pre-trained Model:** Uses a Random Forest Classifier (`model_RFC.sav`) for high accuracy.
- **Custom Dataset Support:** Easily update or extend the dataset for more diseases/symptoms.
- **Performance Metrics:** Visualizations and model evaluation included in the notebook.

---

## 🛠️ Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/vivekchudasama-2004/Disease_Prediction_based_on_symptoms.git
   cd Disease_Prediction_based_on_symptoms
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚡ Usage

### 1. Web Application

Run the web app using:
```bash
python app.py
```
Then open your browser and go to the local URL displayed in your terminal (commonly `http://127.0.0.1:5000/`).

### 2. Jupyter Notebook

1. Open `Disease_Prediction_based_on_symptoms.ipynb` in Jupyter Notebook or JupyterLab.
2. Run the cells to explore the data, train/test models, and make predictions.

---

## 📊 Dataset

- Datasets are stored inside the `Dataset/` directory.
- Format: CSV files with columns representing symptoms (features) and a `disease` label.
- You may add or update datasets to extend the range of diseases and symptoms.

---

## 🤖 Model

- The pre-trained model file is `model_RFC.sav` (Random Forest Classifier).
- The notebook shows steps for data preprocessing, model training, evaluation, and saving/loading the model.

---

## 📈 Results

- Evaluation metrics (accuracy, confusion matrix, etc.) and visualizations are available in the notebook.
- For web predictions, results are displayed directly after submitting symptoms.

---

## 🤝 Contributing

Contributions are very welcome!

1. Fork this repository.
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to your branch: `git push origin feature/YourFeature`
5. Open a pull request describing your changes.

---


*Developed and maintained by [vivekchudasama-2004](https://github.com/vivekchudasama-2004)*
