# India VIX Forecasting Dashboard

## 📘 Project Overview
This project predicts **India VIX (Volatility Index)** using multiple machine learning models. VIX is crucial for assessing market fear and risk, making this tool useful for traders, analysts, and financial researchers.

The project uses:
- Real historical India VIX and NIFTY 50 data
- Advanced feature engineering (lag features, volatility, momentum)
- Five different ML models (Random Forest, Gradient Boosting, Linear, Ridge, and Lasso)
- An interactive Streamlit dashboard for live predictions and visual comparisons

---

## 🖥️ Dashboard Highlights
| Feature | Description |
|--------|-------------|
| 🎯 **Model Selection** | Toggle between different ML models and see how predictions vary |
| 📉 **Real-time Prediction** | Adjust input features using sliders and instantly see VIX prediction |
| 📊 **Metric Display** | Shows RMSE, MAE, R² for each model to understand accuracy |
| 🧮 **Model Insights** | Visual comparison of all models' performance on recent data |
| 🧩 **No Uploads Needed** | All inputs are slider-based — no manual data entry |

---

## 🧠 Important Features Used in Prediction
| Feature Name               | Description |
|---------------------------|-------------|
| `VIX_Close_Lag1/3/5/7`    | Historical VIX close values from 1, 3, 5, and 7 days ago (captures trends) |
| `NIFTY_Returns`           | Daily return of the NIFTY 50 index (market behavior) |
| `VIX_Returns`             | Daily return of the VIX (volatility shift) |
| `VIX_MA_5`, `VIX_STD_5`   | 5-day moving average and standard deviation (trend and noise smoothing) |
| `NIFTY_Volatility`        | 5-day std of NIFTY returns (market instability) |
| `VIX_Rel_Close_Open`      | Measures sentiment change during the day in VIX |
| `VIX_NIFTY_Interaction`   | Captures interplay between market return and volatility changes |
| `VIX_Above_MA`            | Binary: Is VIX currently higher than its 5-day MA? (momentum signal) |
| `NIFTY_Positive`          | Binary: Was today's NIFTY return positive? (market sentiment) |

---

## 📂 Project Structure
```
vix-dashboard/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── data/                 # Data files (CSV, models)
├── models/               # Trained ML models
├── utils/                # Helper functions
└── README.md            # This file
```

## 🚀 How to Run This Project

### 1️⃣ Clone or Download the Repository
```bash
git clone https://github.com/your-repo/vix-dashboard
cd India-VIX-Forecasting-Dashboard
```

### 2️⃣ Set Up Python Environment
```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment on Windows:
myenv\Scripts\activate

# On macOS/Linux:
source myenv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Dashboard
```bash
cd India-VIX-Forecasting-Dashboard
streamlit run main.py
```

### 5️⃣ Access the Dashboard
After running the command, your browser should automatically open to:
```
http://localhost:8501
```

If it doesn't open automatically, copy and paste this URL into your browser.

---


