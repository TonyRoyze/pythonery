# Dynamic Surge Pricing — ML Modeling Pipeline

> Predicting **profit percentage** from ride-share demand/supply dynamics using a systematic, multi-experiment machine learning approach.

---

## Objective

Build a regression model that predicts the **profit percentage** a ride earns over its historical base cost, given real-time demand and supply conditions.

```
profit_percentage = (adjusted_ride_cost − historical_cost) / historical_cost × 100
```

---

## End-to-End Pipeline

```mermaid
flowchart TD
    A([Raw Dataset]) --> B[Data Loading & Inspection]
    B --> C{Issues Found?}
    C -- Duplicates / Missing --> D[Clean & Report]
    C -- None --> E
    D --> E[Feature Engineering\nCreate Target Variable]

    E --> F[profit_percentage\nDerived from demand × supply multipliers]
    F --> G[Feature Group Definition]

    G --> H1[EDA Feature Subset\nRiders, Drivers, Duration\nVehicle_Type, Time_of_Booking]
    G --> H2[Full Feature Set\nAll 5 numerical + 4 categorical]
    G --> H3[SHAP-Selected Features\nNumber_of_Riders, Number_of_Drivers]

    H1 & H2 & H3 --> I[Outlier Analysis]
    I --> J1[No Removal]
    I --> J2[Univariate IQR Removal]
    I --> J3[Multivariate Isolation Forest]

    J1 & J2 & J3 --> K[8 Experiments]
    K --> L[Model Training & Tuning\nGridSearchCV 5-fold CV]
    L --> M[Evaluation\nRMSE · R²]
    M --> N[SHAP Analysis]
    N --> O[Best Model Export\nexported_models_v2/]
    O --> P([HTML Comparison Dashboard])
```

---

## Experiments

```mermaid
flowchart LR
    DS[(Dataset)] --> EDA_NO["Exp 1\nEDA Features\nNo Outlier Removal"]
    DS --> ALL_NO["Exp 2\nAll Features\nNo Outlier Removal"]
    DS --> ALL_UNI["Exp 3\nAll Features\nUnivariate OR"]
    DS --> ALL_MUL["Exp 4\nAll Features\nMultivariate OR (IF)"]
    DS --> EDA_UNI["Exp (i)\nEDA Features\nUnivariate OR"]
    DS --> EDA_MUL["Exp (ii)\nEDA Features\nMultivariate OR (IF)"]
    DS --> SHAP_NB["Exp 5\nSHAP Analysis\non All Features"]
    SHAP_NB --> SHAP_SEL["Exp 6\nSHAP-Selected Features\n+ Extended Models"]

    EDA_NO & ALL_NO & ALL_UNI & ALL_MUL & EDA_UNI & EDA_MUL & SHAP_SEL --> BEST[Best Model per Experiment]
    BEST --> COMPARE[HTML Comparison Table]
```

| # | Experiment | Feature Set | Outlier Treatment | Extended Models |
|---|-----------|-------------|-------------------|:---:|
| 1 | EDA features | `Riders`, `Drivers`, `Duration`, `Vehicle_Type`, `Time_of_Booking` | None | — |
| 2 | All features | Full 5 numerical + 4 categorical | None | — |
| 3 | All features | Full set | Univariate (IQR) | — |
| 4 | All features | Full set | Multivariate (Isolation Forest) | — |
| 5 | SHAP analysis | Full set | None | — |
| 6 | SHAP-selected | `Number_of_Riders`, `Number_of_Drivers` | None | XGBoost, LightGBM, CatBoost |
| (i) | EDA features | EDA subset | Univariate (IQR) | — |
| (ii) | EDA features | EDA subset | Multivariate (Isolation Forest) | — |

---

## Feature Engineering

```mermaid
flowchart TD
    HC[Historical_Cost_of_Ride] & NR[Number_of_Riders] & ND[Number_of_Drivers] --> DM

    subgraph Multipliers
        DM["demand_multiplier\n= Riders / P75  if high demand\n= Riders / P25  if low demand"]
        SM["supply_multiplier\n= P75 / Drivers  if high supply\n= P25 / Drivers  if low supply"]
    end

    DM & SM & HC --> ARC["adjusted_ride_cost\n= Historical × max(demand,0.8) × max(supply,0.8)"]
    ARC & HC --> PP["profit_percentage\n= (adjusted − historical) / historical × 100"]

    PP --> TARGET([Target Variable])
    HC & ARC --> DROP[Dropped from features]
```

---

## Outlier Handling

```mermaid
flowchart LR
    subgraph Univariate ["Univariate — IQR Method"]
        direction TB
        UV1[Compute Q1, Q3, IQR per column]
        UV2["Bounds: Q1 − 1.5×IQR  to  Q3 + 1.5×IQR"]
        UV3[Remove rows outside bounds]
        UV1 --> UV2 --> UV3
    end

    subgraph Multivariate ["Multivariate — Isolation Forest"]
        direction TB
        MV1[Encode categoricals\nScale numericals]
        MV2["IsolationForest\ncontamination = 5%"]
        MV3[Label anomaly = −1]
        MV4[Drop outlier rows from original df]
        MV1 --> MV2 --> MV3 --> MV4
    end

    DF[(DataFrame)] --> Univariate & Multivariate
    Univariate --> C1[Cleaned df — univariate]
    Multivariate --> C2[Cleaned df — multivariate]
```

> **Note:** Isolation Forest is tree-based — scaling has **no effect** on its results. Encoding categoricals is still required for the preprocessor.

---

## Preprocessing Pipeline

```mermaid
flowchart TD
    X[Raw Feature Matrix X] --> CT[ColumnTransformer]

    subgraph CT [ColumnTransformer]
        direction LR
        N["Numerical Features\nStandardScaler"]
        O1["Customer_Loyalty_Status\nOrdinalEncoder\nRegular → Silver → Gold"]
        O2["Vehicle_Type\nOrdinalEncoder\nEconomy → Premium"]
        NH["Nominal Features\nOneHotEncoder\nhandle_unknown='ignore'"]
    end

    CT --> PP[Preprocessed Feature Matrix]
    PP --> MODEL[ML Model]
```

---

## Models Evaluated

```mermaid
mindmap
  root((ML Models))
    Linear
      Linear Regression
      Ridge
      Lasso
      Elastic Net
    Tree-Based
      Decision Tree
      Random Forest
      Gradient Boosting
    Extended Models
      XGBoost
      LightGBM
      CatBoost
```

**Tuning strategy:** `GridSearchCV` with 5-fold `KFold` cross-validation, optimised on **Mean Absolute Error (MAE)**.

---

## Evaluation & Comparison

```mermaid
flowchart LR
    FIT[Fit Pipeline\npreprocessor + model] --> PRED[Predict on train & test]
    PRED --> RMSE["RMSE = √MSE"]
    PRED --> R2["R² score"]
    RMSE & R2 --> TABLE["Styled DataFrame\nbackground_gradient\nRdYlGn_r for RMSE\nRdYlGn for R²"]
    TABLE --> BEST[Identify Best Model\nlowest Test RMSE]
```
---

## SHAP Analysis

```mermaid
flowchart TD
    BM[Best Model from Exp 2\nAll Features] --> EX[shap.TreeExplainer]
    XT[X_train preprocessed] --> EX
    EX --> SV[SHAP Values]

    SV --> SP[Summary Plot\nbeeswarm / dot]
    SV --> DP[Dependence Plot\ntop feature]
    SV --> VI[Variable-level Importance\naggregate one-hot levels]

    VI --> SEL["SHAP-Selected Features\nNumber_of_Riders\nNumber_of_Drivers"]
    SEL --> EXP6[Experiment 6\nSimpler, Faster Model]
```

---

## Key Results Summary

| Experiment | Expected behaviour |
|---|---|
| Exp 2 — All Features | Baseline: highest feature richness |
| Exp 3 / 4 — Outlier Removal | Better generalisation, possibly lower RMSE |
| Exp 5 — SHAP | Reveals top drivers of profit % |
| Exp 6 — SHAP features + extended | Simpler model; extended boosters often win |

> **All experiments share identical `train_test_split(test_size=0.2, random_state=0)` and `KFold(n_splits=5, random_state=42)` to ensure fair comparison.**

