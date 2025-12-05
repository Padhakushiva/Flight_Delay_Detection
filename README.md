# ✈️ Flight Delay Prediction System

A comprehensive machine learning project that predicts flight delays using real-world aviation data, featuring an interactive web dashboard for real-time predictions and analytics.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This project implements a robust machine learning pipeline to predict flight delays with high accuracy. The system includes data preprocessing, feature engineering, model training with class imbalance handling, and an interactive web dashboard for real-time predictions.

### Key Features

- 🤖 **Machine Learning Models**: Logistic Regression, Random Forest, and Decision Tree algorithms
- ⚖️ **Class Imbalance Handling**: SMOTE implementation for balanced training
- 📊 **Interactive Dashboard**: Real-time predictions with comprehensive analytics
- 🔍 **Feature Engineering**: Time-based, route-based, and airline-specific features
- 📈 **Performance Monitoring**: Live model metrics and evaluation
- 🎨 **Professional UI**: Clean, intuitive web interface using Streamlit

## 🚀 Live Demo

Launch the interactive dashboard:

```bash
git clone https://github.com/Padhakushiva/Flight_Delay_Detection
cd Flight_Delay_Detection
pip install -r requirements.txt
streamlit run flight_delay_dashboard.py
```

Visit `http://localhost:8501` to access the dashboard.

## 📊 Dashboard Features

### 🏠 Home & Prediction
- Real-time flight delay predictions
- Interactive input forms for flight details
- Instant probability scores and recommendations

### 📈 Data Analytics
- Comprehensive flight delay statistics
- Route performance analysis
- Airline comparison metrics
- Time-based delay patterns

### 🔍 Model Performance
- Live model accuracy metrics
- Feature importance analysis
- Confusion matrix visualization
- Cross-validation results

### 🗃️ Data Explorer
- Interactive dataset browsing
- Advanced filtering capabilities
- Statistical summaries
- Data quality insights

## 🛠️ Technical Implementation

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Missing value handling
   - Realistic delay simulation (>15 minutes threshold)
   - Data leakage prevention

2. **Feature Engineering**
   - Temporal features (departure hour, time periods)
   - Route-based features (origin-destination pairs)
   - Historical features (airline/route delay rates)
   - Schedule features (weekend/peak operations)

3. **Model Training**
   - Train-test split with stratification
   - SMOTE for class balance handling
   - Cross-validation for robust evaluation
   - Hyperparameter optimization

4. **Model Evaluation**
   - ROC-AUC scoring
   - Precision-Recall analysis
   - Feature importance ranking
   - Confusion matrix analysis

### Key Improvements Made

✅ **Fixed Critical Issues:**
- Corrected delay definition (was using flight duration)
- Removed data leakage (no arrival time in features)
- Implemented realistic delay thresholds
- Added proper temporal and route features

✅ **Enhanced Model Performance:**
- Balanced class distribution with SMOTE
- Cross-validation implementation
- Multiple algorithm comparison
- Feature importance analysis

## 📈 Model Performance

| Model | Accuracy | ROC-AUC | CV Score |
|-------|----------|---------|----------|
| **Logistic Regression** | **96.7%** | **0.892** | **0.885 ± 0.012** |
| Random Forest | 94.2% | 0.878 | 0.871 ± 0.015 |
| Decision Tree | 93.8% | 0.856 | 0.849 ± 0.018 |

### Key Insights

- **Most Important Features**: Departure time (35.9%), Route (28.2%), Airline history (18.9%)
- **Class Distribution**: Balanced 50-50 split after SMOTE implementation
- **Prediction Accuracy**: 99% recall for delay detection
- **Business Value**: Enables proactive passenger management and resource allocation

## 🗂️ Project Structure

```
flight-delay-prediction/
├── 📊 data/
│   └── dashboard_data.csv          # Processed dataset for dashboard
├── 🤖 models/
│   ├── best_model.pkl             # Trained ML model
│   ├── label_encoders.pkl         # Categorical encoders
│   ├── feature_columns.pkl        # Feature definitions
│   ├── model_metrics.pkl          # Performance metrics
│   └── unique_values.pkl          # Dropdown values
├── 📈 flight_delay_dashboard.py    # Main dashboard application
├── 🚀 run_dashboard.py            # Simple launcher script
├── 📓 Untitled.ipynb              # ML pipeline & analysis
├── 📋 requirements.txt            # Package dependencies
├── 📄 Air_full-Raw.csv           # Original dataset
└── 📖 README.md                  # Project documentation
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/[your-username]/flight-delay-prediction.git
   cd flight-delay-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the dashboard**
   ```bash
   streamlit run flight_delay_dashboard.py
   # OR
   python run_dashboard.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - Start making predictions and exploring the data!

## 💡 Usage Examples

### Making Predictions

1. **Select Flight Details:**
   - Choose airline from dropdown
   - Select origin and destination airports
   - Set departure time

2. **Get Instant Results:**
   - Delay probability percentage
   - Risk level classification
   - Confidence interval

3. **Explore Analytics:**
   - View historical delay patterns
   - Compare airline performance
   - Analyze route-specific trends

## 🎓 Educational Value

This project demonstrates:

- **Data Science Workflow**: From raw data to deployed model
- **Machine Learning Best Practices**: Proper train-test splits, cross-validation
- **Class Imbalance Handling**: SMOTE implementation
- **Feature Engineering**: Creating meaningful predictive features
- **Model Evaluation**: Comprehensive performance analysis
- **Web Development**: Interactive dashboard creation
- **Production Deployment**: Model persistence and loading

## 🔮 Future Enhancements

- [ ] Real-time weather data integration
- [ ] Airport congestion information
- [ ] Time-series forecasting features
- [ ] API development for external integration
- [ ] Mobile application
- [ ] A/B testing framework
- [ ] Advanced ensemble methods
- [ ] Automated model retraining pipeline

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Flight data sourced from aviation industry datasets
- Built with scikit-learn, Streamlit, and Plotly
- Inspired by real-world airline operations research

## 📧 Contact

**Project Maintainer**: [Shiva Choudhry]
- GitHub: [@Padhakushiva](https://github.com/Padhakushiva)
- LinkedIn: [Shiva Choudhry](https://www.linkedin.com/in/shivachoudhry/)
- Email: chaudharyshiva2008@example.com

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

*Built with ❤️ and Python*