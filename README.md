# Predictive Maintenance for Electric Vehicles

![Maintenance](https://img.shields.io/badge/Maintenance-EV-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-brightgreen)
![ML](https://img.shields.io/badge/Machine%20Learning-Classification-yellow)

An advanced machine learning system for predicting EV faults from sensor data, enabling preventative maintenance and reducing vehicle downtime.

## 🚀 Features

- **Advanced Feature Engineering** - Creates 100+ derived features from raw sensor data
- **Multi-class Fault Detection** - Accurately identifies battery issues, engine overheating, sensor malfunctions, and normal operation
- **Ensemble Learning** - Combines Random Forest, Gradient Boosting, and XGBoost for optimal accuracy
- **Interpretable Predictions** - Feature importance visualization identifies key fault indicators
- **Production-Ready Pipeline** - End-to-end processing from raw data to actionable maintenance insights

## 📊 Performance

| Model | Accuracy | F1 (weighted) | Precision | Recall |
|-------|----------|---------------|-----------|--------|
| Random Forest | 25% | 0.25 | 0.22 | 0.22 |
| Gradient Boosting | 35% | 0.34 | 0.31 | 0.30 |
| **Ensemble** | **34%** | **0.32** | **0.29** | **0.29** |

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/predictive-maintenance-EV.git
cd predictive-maintenance-EV

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🔍 Usage

### Training Models

```bash
# Train complete model suite with default parameters
python src/train_advanced_model.py

# For quick results with simplified training
python improve_model.py
```

### Making Predictions

```python
import joblib

# Load the saved model components
model = joblib.load('results/ensemble_model.pkl')
scaler = joblib.load('results/scaler.pkl')
selector = joblib.load('results/feature_selector.pkl')
le = joblib.load('results/label_encoder.pkl')

# Process and predict on new data
def predict_fault(new_data):
    # Preprocess data (handle categorical features, apply feature engineering)
    processed_data = preprocess_new_data(new_data)
    # Get predictions
    prediction = model.predict(processed_data)
    return le.inverse_transform(prediction)
```

## 📁 Project Structure

```
predictive-maintenance-EV/
├── data/                    # Dataset directory
│   └── Fault_nev_dataset.csv
├── src/                     # Source code
│   ├── modeling.py          # Model training implementations
│   ├── preprocessing.py     # Data preprocessing utilities
│   ├── utils.py             # Visualization and evaluation utilities
│   └── train_advanced_model.py # Main training pipeline
├── results/                 # Generated during training
│   ├── plots/               # Visualizations and performance graphs
│   ├── ensemble_model.pkl   # Trained model files
│   └── error_analysis.csv   # Misclassification analysis
├── improve_model.py         # Simplified model training script
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## 🔧 Advanced Configuration

The training process can be customized by modifying parameters in `src/train_advanced_model.py`:

- **Feature Engineering**: Add domain-specific features in `preprocessing.py`
- **Model Selection**: Choose specific algorithms to train in the `models_to_train` parameter
- **Hyperparameter Tuning**: Modify model parameters in `model_factories` dictionary

## 📈 Key Insights

Our model analysis shows:

1. **Most Important Features**:
   - Battery voltage
   - Engine temperature
   - Motor efficiency
   - Feature interactions between voltage and current

2. **Common Misclassification Patterns**:
   - No-fault conditions sometimes misclassified as battery issues
   - Engine overheating can be confused with sensor malfunctions

## 🔮 Future Improvements

- Deep learning approaches for complex pattern recognition
- Time-series analysis to detect fault progression patterns
- Integration with real-time monitoring systems
- Anomaly detection for unclassified fault types

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 