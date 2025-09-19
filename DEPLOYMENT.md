# Disease Prediction System - Deployment Guide

This repository contains a Streamlit-based disease prediction application that uses machine learning to predict diseases based on symptoms.

## 🚀 Quick Deploy

### Deploy to Streamlit Cloud
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Set main file path: `Project_main/app.py`
6. Deploy!

### Deploy to Heroku
1. Install [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
2. Clone this repository
3. Login to Heroku: `heroku login`
4. Create a new app: `heroku create your-app-name`
5. Deploy: `git push heroku main`

### Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub account
3. Select this repository
4. Railway will automatically detect and deploy the Streamlit app

## 🛠️ Local Development

### Prerequisites
- Python 3.12+
- pip

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/vivekchudasama-2004/Disease_Prediction_based_on_symptoms.git
   cd Disease_Prediction_based_on_symptoms
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run Project_main/app.py
   ```

## 📂 Project Structure
```
Disease_Prediction_based_on_symptoms/
├── Project_main/
│   ├── app.py                 # Main Streamlit application
│   ├── Dataset/               # Data files
│   ├── Pages/                 # Additional pages
│   ├── model_RFC.sav         # Trained ML model
│   └── requirements.txt      # Project dependencies
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── Procfile                  # Heroku deployment config
├── runtime.txt              # Python version specification
└── README.md                # This file
```

## ⚙️ Configuration

### Environment Variables
Copy `.env.example` to `.env` and configure:
- `HF_USERNAME`: Hugging Face username (optional)
- `HF_TOKEN`: Hugging Face token (optional)
- `DEBUG`: Set to `true` for development

### Streamlit Configuration
The app includes a pre-configured `.streamlit/config.toml` file optimized for deployment.

## 🎯 Features
- **Disease Prediction**: Input symptoms to get disease predictions
- **Model Training**: Train new models with custom datasets
- **Data Upload**: Add new training data
- **Dataset Visualization**: Explore and visualize medical datasets

## 🔧 Troubleshooting

### Common Issues
1. **Module not found errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **Model file missing**: The pre-trained model should be in `Project_main/model_RFC.sav`
3. **Dataset issues**: Ensure all CSV files are present in `Project_main/Dataset/`

### Docker Deployment (Optional)
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "Project_main/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📄 License
This project is open source and available under the MIT License.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.