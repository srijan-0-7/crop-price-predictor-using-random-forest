# 🌾 Crop Price Prediction using Random Forest

This project builds a robust machine learning pipeline to predict crop modal prices based on market, location, season, and temporal features. It leverages real-world agricultural data and visual analytics to uncover pricing patterns and deliver actionable insights for farmers, traders, and policymakers.

---

## 📁 Dataset

- **Source**: `cropds.csv`
- **Contents**: Crop prices across Indian states, districts, and markets over time.
- **Key Columns**:
  - `Commodity_name`
  - `State`, `District`, `Market`
  - `Date`, `Modal Price`

---

## 🧼 Data Preprocessing

- Missing values handled using `na_values='='`
- Outliers removed using **IQR method** for `Modal Price`
- Extracted features from `Date`:
  - `month_column` (e.g., January)
  - `season_names` (e.g., winter, monsoon)
  - `day` (day of week)

---

## 📊 Visualizations

- **Boxplots** to detect price outliers
- **Line plots** to explore seasonal and geographic price trends
- **Bar charts** for district-wise price comparison
- **Heatmap** to show feature correlations

---

## 🔢 Feature Encoding

All categorical variables were numerically encoded using dictionary mapping:
- `Commodity_name`, `State`, `District`, `Market`
- `month_column`, `season_names`

---

## 🧠 Model: Random Forest Regressor

- **Algorithm**: `RandomForestRegressor` from `scikit-learn`
- **Parameters**:
  - `max_depth=1000` for deep learning of complex patterns
  - `random_state=0` for reproducibility
- **Train/Test Split**: 80/20 using `train_test_split`

---

## 📈 Evaluation

- Metric: **R² Score** to measure prediction accuracy
- Output: Predicted modal prices for unseen data

---

## 🧪 Sample Prediction

```python
user_input = [[166, 24, 155, 954, 1, 0, 6]]
regr.predict(user_input)
