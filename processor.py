# process_simple_fix.py - Simple fix for date parsing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_merge_data():
    """Load VIX and NIFTY data and merge them"""
    # Load VIX data
    vix_df = pd.read_csv('data/india_vix.csv')
    
    # Load NIFTY data
    nifty_df = pd.read_csv('data/nifty_50.csv')
    
    # Clean column names
    vix_df.columns = ['Date', 'VIX_Open', 'VIX_High', 'VIX_Low', 'VIX_Close', 'VIX_Prev_Close', 'VIX_Change', 'VIX_Change_Pct']
    nifty_df.columns = ['Index_Name', 'Date', 'NIFTY_Open', 'NIFTY_High', 'NIFTY_Low', 'NIFTY_Close']
    
    # Convert dates - Let pandas auto-detect the format
    print(f"📅 Sample VIX dates: {vix_df['Date'].head(3).tolist()}")
    print(f"📅 Sample NIFTY dates: {nifty_df['Date'].head(3).tolist()}")
    
    # Use pandas automatic date parsing
    vix_df['Date'] = pd.to_datetime(vix_df['Date'])
    nifty_df['Date'] = pd.to_datetime(nifty_df['Date'])
    
    # Merge on date
    df = pd.merge(vix_df, nifty_df[['Date', 'NIFTY_Open', 'NIFTY_High', 'NIFTY_Low', 'NIFTY_Close']], 
                  on='Date', how='inner')
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df

def create_features(df):
    """Create enhanced features for modeling"""
    print("🔧 Creating enhanced features...")

    # 1. Target: Predicting VIX_Close
    df['Overnight_NIFTY_Change_Pct'] = ((df['NIFTY_Open'] - df['NIFTY_Close'].shift(1)) / 
                                        df['NIFTY_Close'].shift(1) * 100)

    df['VIX_Returns'] = df['VIX_Close'].pct_change() * 100
    df['NIFTY_Returns'] = df['NIFTY_Close'].pct_change() * 100

    # 2. Lag Features
    for lag in [1, 3, 5, 7]:
        df[f'VIX_Close_Lag{lag}'] = df['VIX_Close'].shift(lag)
        df[f'NIFTY_Close_Lag{lag}'] = df['NIFTY_Close'].shift(lag)
        df[f'VIX_Returns_Lag{lag}'] = df['VIX_Returns'].shift(lag)
        df[f'NIFTY_Returns_Lag{lag}'] = df['NIFTY_Returns'].shift(lag)

    # 3. Rolling statistics
    df['VIX_MA_5'] = df['VIX_Close'].rolling(window=5).mean()
    df['VIX_MA_10'] = df['VIX_Close'].rolling(window=10).mean()
    df['VIX_STD_5'] = df['VIX_Close'].rolling(window=5).std()
    df['NIFTY_STD_5'] = df['NIFTY_Returns'].rolling(window=5).std()

    # 4. Momentum/Volatility
    df['VIX_Volatility'] = df['VIX_Returns'].rolling(window=5).std()
    df['NIFTY_Volatility'] = df['NIFTY_Returns'].rolling(window=5).std()

    df['VIX_Momentum'] = df['VIX_Close'] - df['VIX_Close'].shift(3)
    df['NIFTY_Momentum'] = df['NIFTY_Close'] - df['NIFTY_Close'].shift(3)

    # 5. Range & Spread Features
    df['VIX_Range'] = df['VIX_High'] - df['VIX_Low']
    df['NIFTY_Range'] = df['NIFTY_High'] - df['NIFTY_Low']
    df['Range_Spread'] = df['VIX_Range'] / (df['NIFTY_Range'] + 1e-6)

    # 6. Price Relative Features
    df['VIX_Rel_Close_Open'] = (df['VIX_Close'] - df['VIX_Open']) / df['VIX_Open']
    df['NIFTY_Rel_Close_Open'] = (df['NIFTY_Close'] - df['NIFTY_Open']) / df['NIFTY_Open']

    # 7. Day/Time Features
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Is_Monday'] = (df['Day_of_Week'] == 0).astype(int)
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter

    # 8. Interaction Features
    df['VIX_NIFTY_Interaction'] = df['VIX_Returns'] * df['NIFTY_Returns']
    df['Lag1_Interaction'] = df['VIX_Returns'].shift(1) * df['NIFTY_Returns'].shift(1)

    # 9. Binarized Indicators
    df['VIX_Above_MA'] = (df['VIX_Close'] > df['VIX_MA_5']).astype(int)
    df['NIFTY_Positive'] = (df['NIFTY_Returns'] > 0).astype(int)

    # Drop NA
    df_clean = df.dropna().reset_index(drop=True)

    # Log summary
    print(f"✅ Enhanced features created! Final shape: {df_clean.shape}")
    print(f"📊 Total usable features: {len([col for col in df_clean.columns if col not in ['Date', 'VIX_Close', 'VIX_Open', 'VIX_High', 'VIX_Low', 'VIX_Prev_Close', 'VIX_Change', 'VIX_Change_Pct', 'NIFTY_Open', 'NIFTY_High', 'NIFTY_Low', 'NIFTY_Close', 'Index_Name']])}")

    return df_clean


def prepare_features_target(df):
    """Prepare features and target for modeling"""
    # Define feature columns (exclude target and non-predictive columns)
    exclude_cols = [
        'Date', 'VIX_Close', 'VIX_Open', 'VIX_High', 'VIX_Low', 'VIX_Prev_Close', 
        'VIX_Change', 'VIX_Change_Pct', 'NIFTY_Open', 'NIFTY_High', 'NIFTY_Low', 'NIFTY_Close', 'Index_Name'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['VIX_Close']
    
    print(f"🎯 Features selected: {len(feature_cols)}")
    print(f"📈 Target: VIX_Close")
    
    return X, y, feature_cols

def calculate_metrics(y_true, y_pred):
    """Calculate RMSE, MAE, and R² metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def train_ridge_lasso_with_cv(X_train, X_test, y_train, y_test):
    """Train Ridge and Lasso Regression using cross-validation"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge_params = {'alpha': [0.1, 1.0, 10.0, 50.0, 100.0]}
    lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}

    # Ridge Regression with CV
    ridge_cv = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_train_scaled, y_train)
    ridge_best = ridge_cv.best_estimator_
    ridge_pred = ridge_best.predict(X_test_scaled)

    # Lasso Regression with CV
    lasso_cv = GridSearchCV(Lasso(max_iter=10000), lasso_params, cv=5, scoring='neg_mean_squared_error')
    lasso_cv.fit(X_train_scaled, y_train)
    lasso_best = lasso_cv.best_estimator_
    lasso_pred = lasso_best.predict(X_test_scaled)

    print("\n🔍 Ridge Regression Best Alpha:", ridge_cv.best_params_['alpha'])
    print("🔍 Lasso Regression Best Alpha:", lasso_cv.best_params_['alpha'])

    return {
        'Ridge Regression': (ridge_best, scaler, ridge_pred),
        'Lasso Regression': (lasso_best, scaler, lasso_pred)
    }


def train_models(X, y):
    """Train multiple models including Ridge and Lasso"""
    print("🚀 Training models...")

    # Time-based train-test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    models = {}
    predictions = {}
    metrics = {}

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    models['Random Forest'] = rf_model
    predictions['Random Forest'] = rf_pred
    metrics['Random Forest'] = calculate_metrics(y_test, rf_pred)

    # Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)

    models['Gradient Boosting'] = gb_model
    predictions['Gradient Boosting'] = gb_pred
    metrics['Gradient Boosting'] = calculate_metrics(y_test, gb_pred)

    # Linear Regression with scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)

    models['Linear Regression'] = (lr_model, scaler)
    predictions['Linear Regression'] = lr_pred
    metrics['Linear Regression'] = calculate_metrics(y_test, lr_pred)

    # Ridge & Lasso with CV
    regularized_models = train_ridge_lasso_with_cv(X_train, X_test, y_train, y_test)

    for name, (model, scaler, pred) in regularized_models.items():
        models[name] = (model, scaler)
        predictions[name] = pred
        metrics[name] = calculate_metrics(y_test, pred)

    # Print results
    print("\n📊 Model Performance:")
    for name, metric in metrics.items():
        print(f"{name}:")
        print(f"  RMSE: {metric['RMSE']:.4f}")
        print(f"  MAE: {metric['MAE']:.4f}")
        print(f"  R²: {metric['R2']:.4f}")
        print()

    return models, predictions, metrics, y_test, X_test

def save_models(models, feature_names):
    """Save models and feature names"""
    with open('models/vix_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("✅ Models saved successfully!")

def load_models():
    """Load saved models"""
    try:
        with open('models/vix_models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        return models, feature_names
    except FileNotFoundError:
        return None, None

def run_complete_pipeline():
    """Run the complete VIX forecasting pipeline"""
    print("🎯 Starting India VIX Forecasting Pipeline...")
    print("="*50)
    
    # Step 1: Load and merge data
    df = load_and_merge_data()
    
    # Step 2: Create features
    df_features = create_features(df)
    
    # Step 3: Prepare features and target
    X, y, feature_names = prepare_features_target(df_features)
    
    # Step 4: Train models
    models, predictions, metrics, y_test, X_test = train_models(X, y)
    
    # Step 5: Save models
    save_models(models, feature_names)
    
    # Return results for Streamlit
    results = {
        'data': df_features,
        'models': models,
        'predictions': predictions,
        'metrics': metrics,
        'y_test': y_test,
        'X_test': X_test,
        'feature_names': feature_names
    }
    
    print("🎉 Pipeline completed successfully!")
    return results

if __name__ == "__main__":
    results = run_complete_pipeline()