# 💉 H1N1 Vaccine Usage Prediction App

A production-ready machine learning application for predicting H1N1 vaccine adoption likelihood based on survey responses.

## Features

✅ **Advanced ML Predictions** - Ensemble learning models with confidence scoring  
✅ **Input Validation** - Comprehensive data validation and error handling  
✅ **Production Infrastructure** - Logging, configuration management, and error handling  
✅ **Professional UI** - Clean, intuitive interface with real-time feedback  
✅ **Scalable Architecture** - Modular code structure for easy maintenance  

## Prerequisites

- Python 3.8+
- pip

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/coderravi0101/Ravi-H1N1-Project.git
cd Ravi-H1N1-Project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Model Setup

### Training Your Model

```python
# In your Jupyter notebook or Python script
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Assuming you have X_train, y_train data
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Save the model
import os
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/h1n1_model.pkl')
```

### Required Directory Structure
```
Ravi-H1N1-Project/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── models/
│   └── h1n1_model.pkl          # Your trained model
├── data/
│   └── (training data files)
├── logs/
│   └── app.log
└── .streamlit/
    └── config.toml
```

## Running the Application

### Development Mode
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Production Deployment

#### Using Streamlit Cloud
1. Push to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click

#### Using Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

## Usage

1. **Fill Survey Information**
   - H1N1 Worry and Awareness levels
   - Doctor recommendations and medical conditions
   - Personal opinions on vaccine effectiveness and risk

2. **Enter Demographics**
   - Age group, education level, gender
   - Household composition

3. **Get Predictions**
   - Receive real-time prediction with confidence score
   - View visual insights and recommendations

## Project Structure

```
app.py              # Main Streamlit application
config.py           # Configuration and constants
requirements.txt    # Python dependencies
.streamlit/         # Streamlit configuration
models/             # Trained ML models
data/               # Training/test data
logs/               # Application logs
```

## Code Quality

- **Type Hints**: Full type annotations for better IDE support
- **Logging**: Comprehensive logging for debugging
- **Error Handling**: Robust exception handling
- **Documentation**: Detailed docstrings and comments
- **Validation**: Input data validation before prediction

## Performance Optimizations

- **Caching**: Model cached for 1 hour (TTL=3600s)
- **Efficient Data Handling**: Vectorized operations with pandas/numpy
- **Minimal Dependencies**: Only essential libraries included

## Troubleshooting

### Model File Not Found
**Error**: "Model file missing" message  
**Solution**: 
1. Train your model following the "Model Setup" section
2. Save it to `models/h1n1_model.pkl`
3. Restart the app

### Import Errors
**Error**: `ModuleNotFoundError`  
**Solution**:
```bash
pip install -r requirements.txt --upgrade
```

### Port Already in Use
**Error**: `Address already in use`  
**Solution**:
```bash
streamlit run app.py --server.port 8502
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project is open source and available under the MIT License.

## Author

**Ravi** - [GitHub Profile](https://github.com/coderravi0101)

## Support

For issues, questions, or suggestions:
- Open an [Issue](https://github.com/coderravi0101/Ravi-H1N1-Project/issues)
- Create a [Discussion](https://github.com/coderravi0101/Ravi-H1N1-Project/discussions)

---

**Last Updated**: August 24, 2026  
**Version**: 1.0.0
