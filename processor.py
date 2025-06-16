# process_simple_fix.py - Simple fix for date parsing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_merge_data():
    """Load VIX and NIFTY data and merge them"""
    # Load VIX data
    vix_df = pd.read_csv('india_vix.csv')
    
    # Load NIFTY data
    nifty_df = pd.read_csv('nifty_50.csv')
    
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
    """Create simplified but effective features for modeling"""
    print("🔧 Creating simplified features...")
    
    # 1. REQUIRED: Overnight NIFTY Change % 
    df['Overnight_NIFTY_Change_Pct'] = ((df['NIFTY_Open'] - df['NIFTY_Close'].shift(1)) / 
                                        df['NIFTY_Close'].shift(1) * 100)
    
    # 2. Basic returns (most important)
    df['VIX_Returns'] = df['VIX_Close'].pct_change() * 100
    df['NIFTY_Returns'] = df['NIFTY_Close'].pct_change() * 100
    
    # 3. Simple lagged features (yesterday's values)
    df['VIX_Yesterday'] = df['VIX_Close'].shift(1)
    df['VIX_Returns_Yesterday'] = df['VIX_Returns'].shift(1)
    df['NIFTY_Returns_Yesterday'] = df['NIFTY_Returns'].shift(1)
    
    # 4. Simple moving averages (only 5-day)
    df['VIX_MA_5'] = df['VIX_Close'].rolling(window=5).mean()
    df['VIX_Above_MA'] = (df['VIX_Close'] > df['VIX_MA_5']).astype(int)  # 1 if above MA, 0 if below
    
    # 5. Basic volatility (5-day only)
    df['NIFTY_Volatility'] = df['NIFTY_Returns'].rolling(window=5).std()
    df['VIX_Volatility'] = df['VIX_Returns'].rolling(window=5).std()
    
    # 6. Simple range features
    df['VIX_Range'] = df['VIX_High'] - df['VIX_Low']
    df['NIFTY_Range'] = df['NIFTY_High'] - df['NIFTY_Low']
    
    # 7. Day of week (simple time feature)
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Is_Monday'] = (df['Day_of_Week'] == 0).astype(int)  # Monday effect
    
    # Drop NaN values
    df_clean = df.dropna().reset_index(drop=True)
    
    print(f"✅ Simplified features created! New shape: {df_clean.shape}")
    print(f"📊 Total features: {len([col for col in df_clean.columns if col not in ['Date', 'VIX_Close', 'VIX_Open', 'VIX_High', 'VIX_Low', 'VIX_Prev_Close', 'VIX_Change', 'VIX_Change_Pct', 'NIFTY_Open', 'NIFTY_High', 'NIFTY_Low', 'NIFTY_Close', 'Index_Name']])}")
    
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

def train_models(X, y):
    """Train multiple models"""
    print("🚀 Training models...")
    
    # Time series split (80-20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    models = {}
    predictions = {}
    metrics = {}
    
    # 1. Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    models['Random Forest'] = rf_model
    predictions['Random Forest'] = rf_pred
    metrics['Random Forest'] = calculate_metrics(y_test, rf_pred)
    
    # 2. Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    
    models['Gradient Boosting'] = gb_model
    predictions['Gradient Boosting'] = gb_pred
    metrics['Gradient Boosting'] = calculate_metrics(y_test, gb_pred)
    
    # 3. Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    
    models['Linear Regression'] = (lr_model, scaler)
    predictions['Linear Regression'] = lr_pred
    metrics['Linear Regression'] = calculate_metrics(y_test, lr_pred)
    
    # Print results
    print("\n📊 Model Performance:")
    for name, metric in metrics.items():
        print(f"{name}:")
        print(f"  RMSE: {metric['RMSE']:.4f}")
        print(f"  MAE: {metric['MAE']:.4f}")
        print(f"  R²: {metric['R2']:.4f}")
        print()
    
    return models, predictions, metrics, y_test, X_test

def calculate_metrics(y_true, y_pred):
    """Calculate model metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def get_feature_importance(model, feature_names, model_name):
    """Get feature importance for tree-based models"""
    if model_name in ['Random Forest', 'Gradient Boosting']:
        importances = model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        return feature_imp.head(15)
    return None

def predict_next_day(models, latest_features, feature_names):
    """Predict next day VIX using all models"""
    predictions = {}
    
    for name, model in models.items():
        try:
            if name == 'Linear Regression':
                lr_model, scaler = model
                scaled_features = scaler.transform([latest_features])
                pred = lr_model.predict(scaled_features)[0]
            else:
                pred = model.predict([latest_features])[0]
            
            predictions[name] = pred
        except Exception as e:
            print(f"Error predicting with {name}: {e}")
            predictions[name] = None
    
    return predictions

def save_models(models, feature_names):
    """Save models and feature names"""
    with open('vix_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("✅ Models saved successfully!")

def load_models():
    """Load saved models"""
    try:
        with open('vix_models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        with open('feature_names.pkl', 'rb') as f:
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