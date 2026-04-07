# ML Pipeline Pro 🚀

A comprehensive, end-to-end Machine Learning pipeline dashboard built with **Streamlit**. This professional web application allows users to upload any tabular dataset (CSV) and walk through a complete data science workflow—from data exploration and cleaning to hyperparameter tuning and performance evaluation—without writing a single line of code.

![App Dashboard Preview](https://via.placeholder.com/800x400.png?text=ML+Pipeline+Pro+Dashboard) <!-- Optional: Add a real screenshot here -->

## ✨ Features

- **📂 Dynamic Data Upload**: Upload any CSV dataset and start analyzing instantly.
- **🔍 Automated EDA**: Instantly generate interactive distributions, correlation matrices, and missing value checks.
- **🛠️ Data Engineering**: Feature imputation (Mean, Median, Mode) and Outlier Detection (IQR, Isolation Forest, DBSCAN, OPTICS).
- **🎯 Feature Selection**: Techniques like Variance Thresholding, Correlation Filtering, and Information Gain.
- **✂️ Data Splitting**: Adjustable train/test splits with stratification support for classification targets.
- **🤖 Model Selection**: Choose between multiple modern algorithms for both **Classification** and **Regression**:
  - Random Forest, Logistic/Linear Regression
  - Support Vector Machines (Linear, RBF, Poly, Sigmoid)
- **🏋️ K-Fold Cross Validation**: Evaluate models robustly across dynamic folds.
- **📊 Performance Metics**: Interactive Confusion Matrices, ROC/Radar charts, Error Residuals, and automated Underfitting/Overfitting diagnosis.
- **⚙️ Hyperparameter Tuning**: Optimize model performance on the fly utilizing Grid Search or Randomized Search mechanisms.

## 🚀 Getting Started

### Prerequisites

You need Python 3.8+ installed. It is recommended to use a virtual environment.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ml-pipeline-pro.git
   cd ml-pipeline-pro
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

To start the dashboard locally:
```bash
streamlit run pipeline.py
```

The application will launch in your default web browser (typically at `http://localhost:8501`).

## 📁 Repository Structure

```text
ml-pipeline-pro/
│
├── pipeline.py          # Main Streamlit application file containing all pipeline tabs and logic
├── requirements.txt     # Python dependency list
├── .gitignore           # Git ignore rules
└── README.md            # This documentation file
```

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) - Frontend dashboard framework
- [Scikit-learn](https://scikit-learn.org/) - Machine learning algorithms and data processing
- [Plotly](https://plotly.com/) - Interactive data visualization
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) - Data manipulation

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
