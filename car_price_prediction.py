# =============================================================================
# Car Price Prediction using Linear Regression & Lasso Regression
# Author: [Your Name]
# Dataset: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_absolute_error


# =============================================================================
# 1. Data Loading
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load the car dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    return df


# =============================================================================
# 2. Data Preprocessing
# =============================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features."""
    df.replace({
        "Fuel_Type":    {'Petrol': 0, 'Diesel': 1, 'CNG': 2},
        "Seller_Type":  {'Dealer': 0, 'Individual': 1},
        "Transmission": {'Manual': 0, 'Automatic': 1},
    }, inplace=True)
    print("Categorical encoding complete.")
    return df


# =============================================================================
# 3. Exploratory Data Analysis
# =============================================================================

def plot_eda(df: pd.DataFrame) -> None:
    """Distribution plots and correlation heatmap."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Car Price Dataset – EDA", fontsize=16)

    sns.histplot(df['Year'], bins=20, ax=axes[0], color='steelblue')
    axes[0].set_title("Car Year Distribution")

    sns.histplot(df['Selling_Price'], bins=30, ax=axes[1], color='salmon')
    axes[1].set_title("Selling Price Distribution")

    sns.histplot(df['Kms_Driven'], bins=30, ax=axes[2], color='seagreen')
    axes[2].set_title("Kms Driven Distribution")

    plt.tight_layout()
    plt.savefig("eda_distributions.png", dpi=150)
    plt.show()
    print("EDA saved as 'eda_distributions.png'")

    # Correlation heatmap (numeric only)
    plt.figure(figsize=(9, 7))
    sns.heatmap(df.select_dtypes(include=np.number).corr(),
                annot=True, fmt='.2f', cmap='Blues', square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.show()
    print("Heatmap saved as 'correlation_heatmap.png'")


# =============================================================================
# 4. Feature / Target Split
# =============================================================================

def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=['Car_Name', 'Selling_Price'], axis=1)
    Y = df['Selling_Price']
    print(f"Features: {X.shape} | Target: {Y.shape}")
    return X, Y


# =============================================================================
# 5. Train / Test Split
# =============================================================================

def split_data(X, Y, test_size=0.1, random_state=1):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, Y_train, Y_test


# =============================================================================
# 6. Model Training
# =============================================================================

def train_models(X_train, Y_train):
    """Train both Linear Regression and Lasso Regression."""
    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)

    lasso_model = Lasso(alpha=0.01)
    lasso_model.fit(X_train, Y_train)

    print("Linear Regression and Lasso training complete.")
    return lin_model, lasso_model


# =============================================================================
# 7. Model Evaluation
# =============================================================================

def evaluate_models(lin_model, lasso_model, X_train, Y_train, X_test, Y_test) -> None:
    """Compare Linear vs Lasso: R² scores + actual vs predicted plots."""
    models = {"Linear Regression": lin_model, "Lasso Regression": lasso_model}

    for name, model in models.items():
        train_preds = model.predict(X_train)
        test_preds  = model.predict(X_test)
        print(f"\n── {name} ──")
        print(f"  Train R²  : {r2_score(Y_train, train_preds):.4f}")
        print(f"  Test  R²  : {r2_score(Y_test,  test_preds):.4f}")
        print(f"  Test  MAE : {mean_absolute_error(Y_test, test_preds):.4f}")

    # Scatter plots: actual vs predicted for both models
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Actual vs Predicted Car Prices", fontsize=14)

    for ax, (name, model) in zip(axes, models.items()):
        test_preds = model.predict(X_test)
        ax.scatter(Y_test, test_preds, alpha=0.6, color='steelblue')
        mn = min(Y_test.min(), test_preds.min())
        mx = max(Y_test.max(), test_preds.max())
        ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect fit')
        ax.set_xlabel("Actual Price (Lakhs)")
        ax.set_ylabel("Predicted Price (Lakhs)")
        ax.set_title(name)
        ax.legend()

    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    plt.show()
    print("Plot saved as 'actual_vs_predicted.png'")


# =============================================================================
# 8. Predictive System
# =============================================================================

def predict_car_price(model, input_data: tuple) -> None:
    """
    Predict selling price for a single car.

    Parameters
    ----------
    input_data : tuple
        (Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type,
         Transmission, Owner)
        Fuel_Type    : Petrol=0, Diesel=1, CNG=2
        Seller_Type  : Dealer=0, Individual=1
        Transmission : Manual=0, Automatic=1
    """
    arr = np.asarray(input_data).reshape(1, -1)
    price = model.predict(arr)[0]
    print(f"\n🚗 Predicted Selling Price: ₹{price:.2f} Lakhs")


# =============================================================================
# Main Pipeline
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = "car data.csv"   # update path if needed

    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    plot_eda(df)

    X, Y = split_features_target(df)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    lin_model, lasso_model = train_models(X_train, Y_train)
    evaluate_models(lin_model, lasso_model, X_train, Y_train, X_test, Y_test)

    # Sample prediction: 2014 car, 6.0 present price, 27000 km, Petrol, Dealer, Manual, 0 owners
    sample = (2014, 6.0, 27000, 0, 0, 0, 0)
    predict_car_price(lasso_model, sample)
