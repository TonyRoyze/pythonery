# 📊 Dynamic Pricing Model for Ride-Sharing

## Reproducibility Steering Document

---

## 1. Problem Definition

### Objectives

1. Identify influential predictors affecting optimal ride fares.
2. Predict **Adjusted Ride Cost** under a dynamic pricing strategy.
3. Predict **Profit Percentage** change when shifting from static → dynamic pricing.

---

## 2. Dataset Description

* Source: Kaggle Dynamic Pricing Dataset
* Observations: **1000**
* Predictors: **10**
* Categorical Predictors: **4**

### Response Variables Created

#### 2.1 Adjusted Cost

Dynamic pricing based on supply–demand conditions.

#### 2.2 Profit Percentage

[
Profit\ Percentage =
\frac{Adjusted\ Cost - Historical\ Cost}{Historical\ Cost} \times 100
]

---

## 3. Data Pre-Processing

### Steps

* Checked for:

  * Missing values → **None found**
  * Duplicate values → **None found**
* Outlier detection → No significant outliers → retained

### Train/Test Split

| Set   | Observations |
| ----- | ------------ |
| Train | 800          |
| Test  | 200          |

### Predictor Adjustment

Removed:

```
Historical_Cost_of_Ride
```

because it was used to construct Adjusted Cost.

---

## 4. Feature Engineering

### Created Variables

| Variable          | Purpose                                       |
| ----------------- | --------------------------------------------- |
| Adjusted_Cost     | Response for dynamic pricing model            |
| Profit_Percentage | Response for pricing transition profitability |

---

## 5. Exploratory Data Analysis (EDA)

---

### 5.1 Distribution of Adjusted Cost

**Plot: Histogram**

Result:

* Left-skewed distribution
* Mean = **680.79**
* Median = **582.77**

---

### 5.2 Scatter Plots vs Adjusted Cost

Generated:

| Predictor              | Result                       |
| ---------------------- | ---------------------------- |
| Number_of_Riders       | No supply curve observed     |
| Number_of_Drivers      | Negative relationship        |
| Average_Ratings        | No clear relationship        |
| Expected_Ride_Duration | Positive linear relationship |

---

### 5.3 Bar Plot

**Mean Adjusted Cost vs Categorical Variables**

Findings:

* Premium vehicle types → higher adjusted cost
* Afternoon bookings → higher adjusted cost
* Location category → no clear distinction
* Customer loyalty → no clear distinction

---

### 5.4 Adjusted Cost vs Historical Cost

**Scatter Plot**

Result:

* Strong linear increase
  (expected because Adjusted Cost is derived from it)

---

### 5.5 Correlation Analysis

Generated:

* Correlation heatmap:

  * Adjusted Cost ↗ with Expected Ride Duration
  * Adjusted Cost ↘ with Number of Drivers
  * Moderate correlation:

    ```
    Number_of_Riders ↔ Number_of_Drivers
    ```

    ⇒ Multicollinearity detected

---

## 6. Profit Percentage Analysis

---

### 6.1 Distribution of Profitability

Result:

| Outcome | Percentage |
| ------- | ---------- |
| Profit  | **83.2%**  |
| Loss    | **16.8%**  |

---

### 6.2 Scatter Plots

| Predictor         | Relationship |
| ----------------- | ------------ |
| Number_of_Riders  | Positive     |
| Number_of_Drivers | Negative     |

---

### 6.3 Kruskal-Wallis Test

Performed between:

```
Categorical Predictors vs Profit_Percentage
```

Result:

* Customer Loyalty Status showed association
* Most other predictors insignificant

---

## 7. Cluster Analysis

### Method Used

* **K-Medoids**
* Distance Metric: **Gower Distance**

### Validation Plots

Generated:

* Average Inertia Plot
* Silhouette Score Plot

Result:

* No inertia flattening
* Increasing silhouette score
* ⇒ **No distinct clusters**

---

## 8. Principal Component Analysis (PCA)

Performed on quantitative variables.

Generated:

* Scree Plot
* Score Plot

Result:

* First two PCs explain ≈ **53% variance**
* No visible clustering in score plot

---

## 9. Modelling – Adjusted Cost

---

### 9.1 Multiple Linear Regression (Forward Selection)

| Dataset | RMSE     | R²     |
| ------- | -------- | ------ |
| Train   | 345.6619 | 0.4957 |
| Test    | 324.2187 | 0.5126 |

---

### 9.2 Residual Diagnostics

| Assumption        | Result              |
| ----------------- | ------------------- |
| Linearity         | Weak                |
| Homoscedasticity  | Violated            |
| Independence      | Violated            |
| Normality         | Violated (Q-Q plot) |
| Multicollinearity | VIF > 10            |

---

### 9.3 Regularization

| Model       | Test RMSE | Test R² |
| ----------- | --------- | ------- |
| Ridge       | 312.9150  | 0.5460  |
| Lasso       | 312.4690  | 0.5473  |
| Elastic Net | 312.5744  | 0.5470  |

Regularization **outperformed MLR**

---

### 9.4 PLS Regression

Optimal Components (CV): **6**

| Test RMSE | Test R² |
| --------- | ------- |
| 313.4026  | 0.5446  |

---

### 9.5 Tree-Based Models

#### Regression Tree

| Train R² | Test R² |
| -------- | ------- |
| 0.8311   | 0.7030  |

---

#### Random Forest

| Train R² | Test R² |
| -------- | ------- |
| 0.9257   | 0.8761  |

Overfitting observed.

---

#### XGBoost

| Train R² | Test R² |
| -------- | ------- |
| 0.9432   | 0.8813  |

Feature Importance:

* Expected Ride Duration
* Number of Riders
* Number of Drivers

Refitting without low-importance predictors → performance decreased
⇒ **Retained all predictors**

✅ **Best Model for Adjusted Cost: XGBoost**

---

## 10. Modelling – Profit Percentage

---

### 10.1 MLR

| Train R² | Test R² |
| -------- | ------- |
| 0.2027   | 0.1624  |

---

### 10.2 Regularization

Best:
Elastic Net → Test R² ≈ **0.2383**

---

### 10.3 PLSR

| Test R² |
| ------- |
| 0.2310  |

---

### 10.4 Regression Tree

| Train R² | Test R² |
| -------- | ------- |
| 0.9806   | 0.9685  |

---

### 10.5 Random Forest

| Train R² | Test R² |
| -------- | ------- |
| 0.9821   | 0.9775  |

Overfitting increased.

---

### 10.6 XGBoost

#### Initial Model

| Train R² | Test R² |
| -------- | ------- |
| 0.9898   | 0.9871  |

Feature Importance:

* Number of Riders
* Number of Drivers

---

#### Refit (Removing Low-Importance Predictors)

| Train R² | Test R²    |
| -------- | ---------- |
| 0.9898   | **0.9893** |

Overfitting reduced.

✅ **Best Model for Profit Percentage: Tuned XGBoost (Reduced Features)**

---

## 11. Final Model Selection

| Target            | Best Model                   |
| ----------------- | ---------------------------- |
| Adjusted Cost     | XGBoost (All Predictors)     |
| Profit Percentage | XGBoost (Reduced Predictors) |

Reason:

* Highest Test R²
* Lowest Test RMSE
* Minimal Train–Test Gap

---

## 12. Deployment Recommendation

Use a hyper-parameter tuned:

```
XGBoost Regressor
```

to predict:

* Adjusted Cost
* Profit Percentage

from input supply–demand and ride characteristics.

---

If you want, I can also turn this into:

* a methods section for your paper
* a workflow checklist
* or a notebook-ready implementation guide

depending on what you’re planning next.
