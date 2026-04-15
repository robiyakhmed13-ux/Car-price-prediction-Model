
# 🚗 Car Price Prediction

A machine learning project that predicts the **selling price of used cars** using **Linear Regression** and **Lasso Regression**, based on features like age, fuel type, and mileage.

---

## 📌 Project Overview

The used car market is massive and pricing a car fairly is hard. This project trains two regression models on CarDekho data and compares their performance to determine the best predictor of selling price.

| Item | Detail |
|------|--------|
| **Algorithms** | Linear Regression, Lasso Regression |
| **Task** | Regression |
| **Dataset** | [Vehicle Dataset – Kaggle (CarDekho)](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) |
| **Target** | `Selling_Price` — price in Lakhs (₹) |

---

## 📂 Project Structure

```
car_price_prediction/
│
├── car_price_prediction.ipynb    # Jupyter Notebook (full walkthrough)
├── car_price_prediction.py       # Clean Python script
├── requirements.txt              # Dependencies
├── car data.csv                  # Dataset (download from Kaggle)
├── eda_distributions.png         # Generated EDA plots
├── correlation_heatmap.png       # Feature correlation heatmap
├── actual_vs_predicted.png       # Actual vs Predicted comparison
└── README.md
```

---

## 📊 Dataset Features

| Feature | Description |
|---------|-------------|
| `Car_Name` | Name of the car model |
| `Year` | Year the car was purchased |
| `Selling_Price` | ✅ **Target** — price the car is being sold for (Lakhs) |
| `Present_Price` | Current ex-showroom price (Lakhs) |
| `Kms_Driven` | Total kilometers driven |
| `Fuel_Type` | Petrol (0) / Diesel (1) / CNG (2) |
| `Seller_Type` | Dealer (0) / Individual (1) |
| `Transmission` | Manual (0) / Automatic (1) |
| `Owner` | Number of previous owners |

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `car data.csv` from [Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) and place it in the project root.

### 4. Run
```bash
python car_price_prediction.py
```

---

## 🔄 Pipeline

```
Raw CSV Data
    │
    ▼
Categorical Encoding (Fuel Type, Seller Type, Transmission)
    │
    ▼
EDA + Correlation Heatmap
    │
    ▼
Train / Test Split (90% / 10%)
    │
    ▼
Linear Regression Training
    │
    ▼
Lasso Regression Training
    │
    ▼
R² + MAE Comparison + Actual vs Predicted Plots
    │
    ▼
Single-car Price Prediction
```

---

## 📈 Results

| Model | Train R² | Test R² |
|-------|----------|---------|
| Linear Regression | ~0.87 | ~0.84 |
| Lasso Regression | ~0.84 | ~0.87 |

> Lasso slightly outperforms Linear Regression on test data due to regularization reducing overfitting.

---

## 🔮 Sample Prediction

```python
# (Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner)
sample = (2014, 6.0, 27000, 0, 0, 0, 0)
predict_car_price(lasso_model, sample)
# Output: 🚗 Predicted Selling Price: ₹X.XX Lakhs
```

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas / numpy** — data processing
- **scikit-learn** — regression models, train/test split, metrics
- **seaborn / matplotlib** — visualization

---

## 🚀 Future Improvements

- [ ] Try Random Forest or XGBoost for non-linear relationships
- [ ] Add more features (car brand, engine size, location)
- [ ] Hyperparameter tuning for Lasso alpha
- [ ] Deploy as a Streamlit price estimator web app

---

## 📄 License

MIT License

---

## 🙋 Author

**[Your Name]**  
[GitHub](https://github.com/your-username) | [LinkedIn](https://linkedin.com/in/your-profile)
