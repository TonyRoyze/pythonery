import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dynamic Pricing — Comprehensive Modeling Notebook

    This notebook combines **all 8** modeling experiments into a single, optimised workflow:

    | # | Experiment | Feature Set | Outlier Treatment |
    |---|-----------|-------------|-------------------|
    | 1 | EDA-selected features | EDA subset | None |
    | 2 | All features | Full set | None |
    | 3 | All features | Full set | Univariate (IQR) |
    | 4 | All features | Full set | Multivariate (Isolation Forest) |
    | 5 | SHAP analysis | Full set | None |
    | 6 | SHAP-selected features | SHAP top 3 | None |
    | (i) | EDA features | EDA subset | Univariate (IQR) |
    | (ii) | EDA features | EDA subset | Multivariate (Isolation Forest) |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Install & Import Libraries
    """)
    return


@app.cell
def _():
    #%pip install kagglehub shap catboost -q
    return


@app.cell
def _():
    import os
    import pathlib
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import shap
    import pickle
    from IPython.display import display, HTML

    from sklearn.model_selection import train_test_split, GridSearchCV, KFold
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer

    from catboost import CatBoostRegressor

    return (
        CatBoostRegressor,
        ColumnTransformer,
        DecisionTreeRegressor,
        ElasticNet,
        GradientBoostingRegressor,
        GridSearchCV,
        IsolationForest,
        KFold,
        Lasso,
        LinearRegression,
        OneHotEncoder,
        OrdinalEncoder,
        Pipeline,
        RandomForestRegressor,
        Ridge,
        StandardScaler,
        display,
        make_scorer,
        mean_absolute_error,
        mean_squared_error,
        np,
        pd,
        pickle,
        plt,
        r2_score,
        shap,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Data Loading & Initial Inspection
    """)
    return


@app.cell
def _(pd):
    # import kagglehub
    # path = kagglehub.dataset_download("arashnic/dynamic-pricing-dataset")
    # df = pd.read_csv(os.path.join(path, "dynamic_pricing.csv"))
    df = pd.read_csv( "dynamic_pricing.csv")
    return (df,)


@app.cell
def _(df):
    # Basic data inspection
    print(f'Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns')
    print(f'Duplicates: {df.duplicated().sum()}')
    df.info()
    return


@app.cell
def _(df, display):
    # Missing values
    missing = df.isna().sum().reset_index()
    missing.columns = ['features', 'missing_count']
    missing['percentage'] = missing['missing_count'] / df.shape[0] * 100
    missing_only = missing[missing['missing_count'] > 0].sort_values(by='missing_count', ascending=False).reset_index(drop=True)
    display(missing_only.style.background_gradient()) if len(missing_only) > 0 else print('No missing values found.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Create Target Variable (`adjusted_ride_cost`)

    The adjusted ride cost is derived from the historical cost amplified by demand and supply multipliers:

    ```
    adjusted_ride_cost = Historical_Cost_of_Ride × max(demand_mult, 0.8) × max(supply_mult, 0.8)
    ```
    """)
    return


@app.cell
def _(df, np):
    # --- Demand multiplier ---
    high_demand_percentile = 75
    low_demand_percentile = 25
    df['demand_multiplier'] = np.where(df['Number_of_Riders'] > np.percentile(df['Number_of_Riders'], high_demand_percentile), df['Number_of_Riders'] / np.percentile(df['Number_of_Riders'], high_demand_percentile), df['Number_of_Riders'] / np.percentile(df['Number_of_Riders'], low_demand_percentile))
    high_supply_percentile = 75
    low_supply_percentile = 25
    df['supply_multiplier'] = np.where(df['Number_of_Drivers'] > np.percentile(df['Number_of_Drivers'], low_supply_percentile), np.percentile(df['Number_of_Drivers'], high_supply_percentile) / df['Number_of_Drivers'], np.percentile(df['Number_of_Drivers'], low_supply_percentile) / df['Number_of_Drivers'])
    demand_threshold_high = 1.2
    demand_threshold_low = 0.8
    supply_threshold_high = 0.8
    # --- Supply multiplier ---
    supply_threshold_low = 1.2
    df['adjusted_ride_cost'] = df['Historical_Cost_of_Ride'] * (np.maximum(df['demand_multiplier'], demand_threshold_low) * np.maximum(df['supply_multiplier'], supply_threshold_high))
    df_1 = df.drop(['demand_multiplier', 'supply_multiplier', 'Historical_Cost_of_Ride'], axis=1)
    print('After target creation:', df_1.shape)
    # --- Thresholds ---
    # --- Adjusted cost ---
    # --- Drop helper columns ---
    df_1.info()
    return (df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. Feature Groups

    Define feature splits used across all experiments.
    """)
    return


@app.cell
def _():
    # ---- Column groups ----
    all_cat_cols  = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']
    all_num_cols  = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides',
                     'Average_Ratings', 'Expected_Ride_Duration']

    # Full feature set (Experiments 2, 3, 4, 5)
    numerical_features = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides',
                          'Average_Ratings', 'Expected_Ride_Duration']
    ordinal_features   = ['Customer_Loyalty_Status', 'Vehicle_Type']
    nominal_features   = ['Location_Category', 'Time_of_Booking']

    # EDA-selected feature subset (Experiments 1, (i), (ii))
    eda_numerical_features = ['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration']
    eda_ordinal_features   = ['Vehicle_Type']
    eda_nominal_features   = ['Time_of_Booking']

    # SHAP-selected features (Experiment 6)
    shap_selected_numerical = ['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration']
    return (
        all_num_cols,
        eda_nominal_features,
        eda_numerical_features,
        eda_ordinal_features,
        nominal_features,
        numerical_features,
        ordinal_features,
        shap_selected_numerical,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5. Shared Utilities

    Common functions shared across all experiments — defined once to avoid repetition.
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    Pipeline,
    StandardScaler,
):
    def build_preprocessor(num_feats, ord_feats, nom_feats):
        """Build a ColumnTransformer preprocessor.
    
        Args:
            num_feats: List of numerical feature names (will be StandardScaled).
            ord_feats: List of ordinal feature names. Handles 'Customer_Loyalty_Status' and 'Vehicle_Type'.
            nom_feats: List of nominal feature names (will be OneHotEncoded).
        """
        numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        nominal_transformer   = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        encoder_loy = OrdinalEncoder(categories=[['Regular', 'Silver', 'Gold']])
        encoder_veh = OrdinalEncoder(categories=[['Economy', 'Premium']])

        transformers = [('num', numerical_transformer, num_feats)]
        if 'Customer_Loyalty_Status' in ord_feats:
            transformers.append(('loyl', encoder_loy, ['Customer_Loyalty_Status']))
        if 'Vehicle_Type' in ord_feats:
            transformers.append(('vehi', encoder_veh, ['Vehicle_Type']))
        if nom_feats:
            transformers.append(('nomi', nominal_transformer, nom_feats))

        return ColumnTransformer(transformers=transformers)

    return (build_preprocessor,)


@app.cell
def _(
    CatBoostRegressor,
    DecisionTreeRegressor,
    ElasticNet,
    GradientBoostingRegressor,
    KFold,
    Lasso,
    LinearRegression,
    RandomForestRegressor,
    Ridge,
    make_scorer,
    mean_absolute_error,
):
    # Cross-validation & scoring strategy
    cv         = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)


    def build_models_config(include_catboost=False):
        """Return a dict of model name -> {model, params} for experiment runs."""
        config = {
            "Linear Regression": {"model": LinearRegression(), "params": None},
            "Ridge": {
                "model": Ridge(random_state=42),
                "params": {"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
            },
            "Lasso": {
                "model": Lasso(random_state=42, max_iter=2000),
                "params": {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
            },
            "Elastic Net": {
                "model": ElasticNet(random_state=42, max_iter=2000),
                "params": {
                    "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
                    "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                },
            },
            "Decision Tree": {
                "model": DecisionTreeRegressor(random_state=42),
                "params": {
                    "model__max_depth": [5, 10, 15, 20, None],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4],
                },
            },
            "Random Forest": {
                "model": RandomForestRegressor(random_state=42, n_jobs=-1),
                "params": {
                    "model__n_estimators": [50, 100, 200],
                    "model__max_depth": [10, 20, None],
                    "model__min_samples_split": [2, 5, 10],
                },
            },
            "Gradient Boosting": {
                "model": GradientBoostingRegressor(random_state=42),
                "params": {
                    "model__n_estimators": [50, 100, 200],
                    "model__learning_rate": [0.01, 0.1, 0.2],
                    "model__max_depth": [3, 5, 7],
                },
            },
        }
        if include_catboost:
            config["CatBoost"] = {
                "model": CatBoostRegressor(random_state=42, verbose=0),
                "params": {
                    "model__iterations": [200, 500, 800],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__depth": [4, 6, 8],
                    "model__l2_leaf_reg": [1, 3, 5, 7],
                },
            }
        return config

    return build_models_config, cv, mae_scorer


@app.cell
def _(
    GridSearchCV,
    Pipeline,
    cv,
    mae_scorer,
    mean_squared_error,
    np,
    pd,
    plt,
    r2_score,
    sns,
):
    def run_experiment(X_train, X_test, y_train, y_test, preprocessor, models_config):
        """Train, tune and evaluate all models.
    
        Returns:
            results      : dict of name -> metrics + pipeline
            best_name    : name of the best model (lowest Test RMSE)
            best_model   : fitted pipeline of the best model
        """
        results = {}
        for name, config in models_config.items():
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", config["model"])])

            if config["params"] is not None:
                print(f"  Tuning {name}...")
                gs = GridSearchCV(pipeline, config["params"], cv=cv, scoring=mae_scorer, n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)
                best_pipeline = gs.best_estimator_
                print(f"    Best params: {gs.best_params_}")
            else:
                best_pipeline = pipeline
                best_pipeline.fit(X_train, y_train)

            y_train_pred = best_pipeline.predict(X_train)
            y_test_pred  = best_pipeline.predict(X_test)

            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_r2   = r2_score(y_train, y_train_pred)
            test_rmse  = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_r2    = r2_score(y_test, y_test_pred)

            results[name] = {
                "Train_RMSE": train_rmse, "Train_R2": train_r2,
                "Test_RMSE":  test_rmse,  "Test_R2":  test_r2,
                "pipeline":   best_pipeline,
            }
            print(f"  {name}: Train ${train_rmse:.2f} / R2={train_r2:.4f}  |  Test ${test_rmse:.2f} / R2={test_r2:.4f}")

        comparison_df = pd.DataFrame.from_dict(results, orient='index')[['Train_RMSE','Train_R2','Test_RMSE','Test_R2']]
        best_name     = comparison_df['Test_RMSE'].idxmin()
        best_model    = results[best_name]['pipeline']

        print(f"\n>>> Best model: {best_name}")
        print(f"    Test RMSE: ${results[best_name]['Test_RMSE']:.2f}  |  R2: {results[best_name]['Test_R2']:.4f}")
        print("\nModel Comparison:")
        print(comparison_df.drop(columns='pipeline', errors='ignore').to_string())
        return results, best_name, best_model


    def plot_predictions(y_test, y_pred, model_name, label=''):
        """Side-by-side: Actual vs Predicted scatter + Residuals plot."""
        residuals = y_test - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=axes[0], color='steelblue')
        max_val = max(y_test.max(), y_pred.max())
        min_val = min(y_test.min(), y_pred.min())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0].set_title(f'{model_name} ({label}): Actual vs. Predicted', fontsize=13)
        axes[0].set_xlabel('Actual Adjusted Ride Cost')
        axes[0].set_ylabel('Predicted Adjusted Ride Cost')

        sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=axes[1], color='teal')
        axes[1].axhline(y=0, color='red', linestyle='--', lw=2)
        axes[1].set_title(f'{model_name} ({label}): Residuals', fontsize=13)
        axes[1].set_xlabel('Predicted Adjusted Ride Cost')
        axes[1].set_ylabel('Residuals (Actual - Predicted)')

        plt.tight_layout()
        plt.show()

    return plot_predictions, run_experiment


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 6. Outlier Detection & Removal Utilities
    """)
    return


@app.cell
def _(IsolationForest, build_preprocessor, pd):
    def detect_univariate_outliers(df, cols):
        """Report IQR-based outliers for each numeric column."""
        n_rows = len(df)
        rows = []
        for col in cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            n_out = ((df[col] < lb) | (df[col] > ub)).sum()
            rows.append({'Feature': col, 'Lower': round(lb, 2), 'Upper': round(ub, 2),
                         'Outliers': n_out, 'Pct (%)': round(n_out / n_rows * 100, 2)})
        return pd.DataFrame(rows)


    def remove_univariate_outliers(df, cols):
        """Remove IQR-based outliers for all specified columns (iterative)."""
        df_clean = df.copy()
        print(f'Original shape: {df_clean.shape}')
        for col in cols:
            Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            before = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lb) & (df_clean[col] <= ub)]
            print(f'  {col}: removed {before - len(df_clean)} rows -> shape {df_clean.shape}')
        return df_clean


    def remove_multivariate_outliers_iso_forest(df, num_feats, ord_feats, nom_feats,
                                                 contamination=0.05, random_state=42):
        """
        Detect and remove multivariate outliers using Isolation Forest.
    
        Note: Isolation Forest is tree-based so scaling has no impact,
        but categorical encoding is still needed.
    
        Returns:
            df_clean       : DataFrame with outliers removed
            outlier_indexes: List of dropped row indices
        """
        df_copy = df.copy()
        prep = build_preprocessor(num_feats, ord_feats, nom_feats)
        X_prepared = prep.fit_transform(df_copy)

        iso = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
        df_copy['anomaly_label'] = iso.fit_predict(X_prepared)

        outlier_indexes = df_copy[df_copy['anomaly_label'] == -1].index.tolist()
        print(f'Outliers: {len(outlier_indexes)} ({len(outlier_indexes)/len(df)*100:.1f}% of data)')
        print(f'Clean data: {len(df) - len(outlier_indexes)} rows')

        return df.drop(index=outlier_indexes), outlier_indexes

    return (
        detect_univariate_outliers,
        remove_multivariate_outliers_iso_forest,
        remove_univariate_outliers,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 7. Experiment 1 — EDA-Selected Features (No Outlier Removal)

    **Features:** `Number_of_Riders`, `Number_of_Drivers`, `Expected_Ride_Duration`, `Vehicle_Type`, `Time_of_Booking`
    """)
    return


@app.cell
def _(
    build_models_config,
    build_preprocessor,
    df_1,
    eda_nominal_features,
    eda_numerical_features,
    eda_ordinal_features,
    run_experiment,
    train_test_split,
):
    print('=' * 60)
    print('EXPERIMENT 1: EDA-Selected Features — No Outlier Removal')
    print('=' * 60)
    X_e1 = df_1[eda_ordinal_features + eda_nominal_features + eda_numerical_features]
    y_e1 = df_1['adjusted_ride_cost']
    X_train_e1, X_test_e1, y_train_e1, y_test_e1 = train_test_split(X_e1, y_e1, test_size=0.2, random_state=0)
    preprocessor_e1 = build_preprocessor(eda_numerical_features, eda_ordinal_features, eda_nominal_features)
    models_config_e1 = build_models_config(include_catboost=False)
    results_e1, best_name_e1, best_model_e1 = run_experiment(X_train_e1, X_test_e1, y_train_e1, y_test_e1, preprocessor_e1, models_config_e1)
    return X_test_e1, best_model_e1, best_name_e1, y_test_e1


@app.cell
def _(X_test_e1, best_model_e1, best_name_e1, plot_predictions, y_test_e1):
    y_pred_e1 = best_model_e1.predict(X_test_e1)
    plot_predictions(y_test_e1, y_pred_e1, best_name_e1, label='EDA features')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 8. Experiment 2 — All Features (No Outlier Removal)

    **Features:** All numerical + ordinal + nominal columns.
    """)
    return


@app.cell
def _(
    build_models_config,
    build_preprocessor,
    df_1,
    nominal_features,
    numerical_features,
    ordinal_features,
    run_experiment,
    train_test_split,
):
    print('=' * 60)
    print('EXPERIMENT 2: All Features — No Outlier Removal')
    print('=' * 60)
    X_e2 = df_1[ordinal_features + nominal_features + numerical_features]
    y_e2 = df_1['adjusted_ride_cost']
    X_train_e2, X_test_e2, y_train_e2, y_test_e2 = train_test_split(X_e2, y_e2, test_size=0.2, random_state=0)
    preprocessor_e2 = build_preprocessor(numerical_features, ordinal_features, nominal_features)
    models_config_e2 = build_models_config(include_catboost=False)
    results_e2, best_name_e2, best_model_e2 = run_experiment(X_train_e2, X_test_e2, y_train_e2, y_test_e2, preprocessor_e2, models_config_e2)
    return X_test_e2, X_train_e2, best_model_e2, best_name_e2, y_test_e2


@app.cell
def _(X_test_e2, best_model_e2, best_name_e2, plot_predictions, y_test_e2):
    y_pred_e2 = best_model_e2.predict(X_test_e2)
    plot_predictions(y_test_e2, y_pred_e2, best_name_e2, label='All features')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 9. Univariate Outlier Analysis (IQR Method)

    Detect outliers in numerical columns before applying removal.
    """)
    return


@app.cell
def _(all_num_cols, detect_univariate_outliers, df_1):
    outlier_report = detect_univariate_outliers(df_1, all_num_cols)
    print(outlier_report.to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9a. Experiment 3 — All Features + Univariate Outlier Removal
    """)
    return


@app.cell
def _(
    all_num_cols,
    build_models_config,
    build_preprocessor,
    df_1,
    nominal_features,
    numerical_features,
    ordinal_features,
    remove_univariate_outliers,
    run_experiment,
    train_test_split,
):
    print('=' * 60)
    print('EXPERIMENT 3: All Features — Univariate Outliers Removed')
    print('=' * 60)
    df_e3 = remove_univariate_outliers(df_1, all_num_cols)
    X_e3 = df_e3[ordinal_features + nominal_features + numerical_features]
    y_e3 = df_e3['adjusted_ride_cost']
    X_train_e3, X_test_e3, y_train_e3, y_test_e3 = train_test_split(X_e3, y_e3, test_size=0.2, random_state=0)
    preprocessor_e3 = build_preprocessor(numerical_features, ordinal_features, nominal_features)
    models_config_e3 = build_models_config(include_catboost=False)
    results_e3, best_name_e3, best_model_e3 = run_experiment(X_train_e3, X_test_e3, y_train_e3, y_test_e3, preprocessor_e3, models_config_e3)
    return X_test_e3, best_model_e3, best_name_e3, y_test_e3


@app.cell
def _(X_test_e3, best_model_e3, best_name_e3, plot_predictions, y_test_e3):
    y_pred_e3 = best_model_e3.predict(X_test_e3)
    plot_predictions(y_test_e3, y_pred_e3, best_name_e3, label='All features + Univariate OR')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9b. Experiment (i) — EDA Features + Univariate Outlier Removal
    """)
    return


@app.cell
def _(
    build_models_config,
    build_preprocessor,
    df_1,
    eda_nominal_features,
    eda_numerical_features,
    eda_ordinal_features,
    remove_univariate_outliers,
    run_experiment,
    train_test_split,
):
    print('=' * 60)
    print('EXPERIMENT (i): EDA Features — Univariate Outliers Removed')
    print('=' * 60)
    eda_num_cols_for_or = eda_numerical_features + ['adjusted_ride_cost']
    # Use EDA numerical + adjusted_ride_cost for outlier detection scope
    df_ei = remove_univariate_outliers(df_1, eda_num_cols_for_or)
    X_ei = df_ei[eda_ordinal_features + eda_nominal_features + eda_numerical_features]
    y_ei = df_ei['adjusted_ride_cost']
    X_train_ei, X_test_ei, y_train_ei, y_test_ei = train_test_split(X_ei, y_ei, test_size=0.2, random_state=0)
    preprocessor_ei = build_preprocessor(eda_numerical_features, eda_ordinal_features, eda_nominal_features)
    models_config_ei = build_models_config(include_catboost=False)
    results_ei, best_name_ei, best_model_ei = run_experiment(X_train_ei, X_test_ei, y_train_ei, y_test_ei, preprocessor_ei, models_config_ei)
    return X_test_ei, best_model_ei, best_name_ei, y_test_ei


@app.cell
def _(X_test_ei, best_model_ei, best_name_ei, plot_predictions, y_test_ei):
    y_pred_ei = best_model_ei.predict(X_test_ei)
    plot_predictions(y_test_ei, y_pred_ei, best_name_ei, label='EDA features + Univariate OR')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 10. Multivariate Outlier Detection — Isolation Forest

    ## 10a. Experiment 4 — All Features + Multivariate Outlier Removal

    > **Note:** Isolation Forest is tree-based; scaling has no effect on results.
    """)
    return


@app.cell
def _(
    build_models_config,
    build_preprocessor,
    df_1,
    nominal_features,
    numerical_features,
    ordinal_features,
    remove_multivariate_outliers_iso_forest,
    run_experiment,
    train_test_split,
):
    print('=' * 60)
    print('EXPERIMENT 4: All Features — Multivariate Outliers Removed')
    print('=' * 60)
    df_e4, outlier_idx_e4 = remove_multivariate_outliers_iso_forest(df_1, numerical_features, ordinal_features, nominal_features)
    X_e4 = df_e4[ordinal_features + nominal_features + numerical_features]
    y_e4 = df_e4['adjusted_ride_cost']
    X_train_e4, X_test_e4, y_train_e4, y_test_e4 = train_test_split(X_e4, y_e4, test_size=0.2, random_state=0)
    preprocessor_e4 = build_preprocessor(numerical_features, ordinal_features, nominal_features)
    models_config_e4 = build_models_config(include_catboost=False)
    results_e4, best_name_e4, best_model_e4 = run_experiment(X_train_e4, X_test_e4, y_train_e4, y_test_e4, preprocessor_e4, models_config_e4)
    return X_test_e4, best_model_e4, best_name_e4, y_test_e4


@app.cell
def _(X_test_e4, best_model_e4, best_name_e4, plot_predictions, y_test_e4):
    y_pred_e4 = best_model_e4.predict(X_test_e4)
    plot_predictions(y_test_e4, y_pred_e4, best_name_e4, label='All features + Multivariate OR')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10b. Experiment (ii) — EDA Features + Multivariate Outlier Removal
    """)
    return


@app.cell
def _(
    build_models_config,
    build_preprocessor,
    df_1,
    eda_nominal_features,
    eda_numerical_features,
    eda_ordinal_features,
    remove_multivariate_outliers_iso_forest,
    run_experiment,
    train_test_split,
):
    print('=' * 60)
    print('EXPERIMENT (ii): EDA Features — Multivariate Outliers Removed')
    print('=' * 60)
    df_eii, outlier_idx_eii = remove_multivariate_outliers_iso_forest(df_1, eda_numerical_features, eda_ordinal_features, eda_nominal_features)
    X_eii = df_eii[eda_ordinal_features + eda_nominal_features + eda_numerical_features]
    y_eii = df_eii['adjusted_ride_cost']
    X_train_eii, X_test_eii, y_train_eii, y_test_eii = train_test_split(X_eii, y_eii, test_size=0.2, random_state=0)
    preprocessor_eii = build_preprocessor(eda_numerical_features, eda_ordinal_features, eda_nominal_features)
    models_config_eii = build_models_config(include_catboost=False)
    results_eii, best_name_eii, best_model_eii = run_experiment(X_train_eii, X_test_eii, y_train_eii, y_test_eii, preprocessor_eii, models_config_eii)
    return X_test_eii, best_model_eii, best_name_eii, y_test_eii


@app.cell
def _(X_test_eii, best_model_eii, best_name_eii, plot_predictions, y_test_eii):
    y_pred_eii = best_model_eii.predict(X_test_eii)
    plot_predictions(y_test_eii, y_pred_eii, best_name_eii, label='EDA features + Multivariate OR')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 11. SHAP Feature Importance Analysis

    Using the best model from **Experiment 2** (All Features, No Outlier Removal) for SHAP analysis.
    """)
    return


@app.cell
def _(best_model_e2, best_name_e2):
    # Extract components from Experiment 2's best model
    preprocessor_shap  = best_model_e2.named_steps['preprocessor']
    model_shap         = best_model_e2.named_steps['model']
    feature_names_shap = preprocessor_shap.get_feature_names_out()

    print(f'Best model for SHAP: {best_name_e2}')
    print(f'Features after preprocessing: {len(feature_names_shap)}')
    return feature_names_shap, model_shap, preprocessor_shap


@app.cell
def _(best_name_e2, feature_names_shap, model_shap, pd):
    # Sklearn feature importances (if available)
    if hasattr(model_shap, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names_shap,
            'Importance': model_shap.feature_importances_
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

        print(f'Top 15 Feature Importances ({best_name_e2}):')
        print(importance_df.head(15).to_string(index=False))
    else:
        print('Model does not expose feature_importances_.')
    return


@app.cell
def _(X_train_e2, feature_names_shap, model_shap, pd, preprocessor_shap, shap):
    # Preprocess training data for SHAP
    X_train_prep = preprocessor_shap.transform(X_train_e2)
    if hasattr(X_train_prep, 'toarray'):
        X_train_prep = X_train_prep.toarray()
    X_train_prep_df = pd.DataFrame(X_train_prep, columns=feature_names_shap)

    explainer   = shap.TreeExplainer(model_shap)
    shap_values = explainer.shap_values(X_train_prep_df)

    print('SHAP values computed.')
    return X_train_prep_df, shap_values


@app.cell
def _(X_train_prep_df, feature_names_shap, plt, shap, shap_values):
    # SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train_prep_df, plot_type='dot',
                      feature_names=feature_names_shap, show=False)
    plt.title('SHAP Summary Plot — Feature Impact on Adjusted Ride Cost')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X_train_prep_df, feature_names_shap, np, plt, shap, shap_values):
    # SHAP Dependence Plot for the top feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_feat_idx  = np.argmax(mean_abs_shap)
    top_feat_name = feature_names_shap[top_feat_idx]
    print(f'Most important feature: {top_feat_name}')

    plt.figure(figsize=(10, 6))
    shap.dependence_plot(top_feat_name, shap_values, X_train_prep_df,
                         interaction_index='auto', show=False)
    plt.title(f'SHAP Dependence Plot — {top_feat_name}')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    X_train_e2,
    feature_names_shap,
    nominal_features,
    np,
    numerical_features,
    ordinal_features,
    pd,
    shap_values,
):
    # Aggregate SHAP importance at variable level (handles one-hot expansion)
    shap_df_all = pd.DataFrame(shap_values, columns=feature_names_shap, index=X_train_e2.index)

    shap_var_importance = {}

    for var in nominal_features:
        cols = [c for c in feature_names_shap if c.startswith(f'nomi__{var}')]
        if cols:
            shap_var_importance[var] = np.abs(shap_df_all[cols]).mean().sum()

    for var in ordinal_features:
        cols = [c for c in feature_names_shap if var in c]
        if cols:
            shap_var_importance[var] = np.abs(shap_df_all[cols]).mean().sum()

    for var in numerical_features:
        cols = [c for c in feature_names_shap if f'num__{var}' in c or c == var]
        if cols:
            shap_var_importance[var] = np.abs(shap_df_all[cols]).mean().sum()

    shap_var_df = pd.DataFrame({
        'Variable': list(shap_var_importance.keys()),
        'Mean_Absolute_SHAP': list(shap_var_importance.values())
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False).reset_index(drop=True)

    print(shap_var_df.to_string(index=False))
    return (shap_var_df,)


@app.cell
def _(plt, shap_var_df, sns):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=shap_var_df, x='Mean_Absolute_SHAP', y='Variable', palette='viridis')
    plt.title('Variable-wise SHAP Importance')
    plt.xlabel('Mean Absolute SHAP Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 12. Experiment 6 — Modeling with SHAP-Selected Features

    SHAP identified the top 3 numeric drivers: `Number_of_Riders`, `Number_of_Drivers`, `Expected_Ride_Duration`.

    This experiment tests whether a simpler model with only these features is competitive.
    CatBoost is included here as in the original notebook.
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    StandardScaler,
    build_models_config,
    df_1,
    run_experiment,
    shap_selected_numerical,
    train_test_split,
):
    print('=' * 60)
    print('EXPERIMENT 6: SHAP-Selected Features')
    print('=' * 60)
    X_e6 = df_1[shap_selected_numerical]
    y_e6 = df_1['adjusted_ride_cost']
    X_train_e6, X_test_e6, y_train_e6, y_test_e6 = train_test_split(X_e6, y_e6, test_size=0.2, random_state=0)
    preprocessor_e6 = ColumnTransformer(transformers=[('num', StandardScaler(), shap_selected_numerical)])
    models_config_e6 = build_models_config(include_catboost=False)
    # Numerical-only preprocessor (no ordinal/nominal needed)
    results_e6, best_name_e6, best_model_e6 = run_experiment(X_train_e6, X_test_e6, y_train_e6, y_test_e6, preprocessor_e6, models_config_e6)
    return X_test_e6, best_model_e6, best_name_e6, y_test_e6


@app.cell
def _(X_test_e6, best_model_e6, best_name_e6, plot_predictions, y_test_e6):
    y_pred_e6 = best_model_e6.predict(X_test_e6)
    plot_predictions(y_test_e6, y_pred_e6, best_name_e6, label='SHAP-selected features')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 13. Export Best Models

    Each experiment's best pipeline (preprocessor + model) is saved to `exported_models/` as a pickle file so it can be loaded and used in production without re-training.
    """)
    return


@app.cell
def _(best_model_e6, pickle):
    models_to_export = {'exp1.6_shap_features': best_model_e6}
    for model_key, pipeline in models_to_export.items():  # 'exp1.1_eda_features':          best_model_e1,
        out_path = f'./{model_key}.pkl'  # 'exp1.2_all_features':          best_model_e2,
        with open(out_path, 'wb') as _f:  # 'exp1.3_all_univariate_or':     best_model_e3,
            pickle.dump(pipeline, _f)  # 'exp1.4_all_multivariate_or':   best_model_e4,
        print(f'Saved: {out_path}')  # 'exp1.i_eda_univariate_or':     best_model_e1_i,  # 'exp1.ii_eda_multivariate_or':  best_model_e1_ii,
    return


@app.cell
def _(X_test_e2, best_model_e2, np, pickle):
    # Quick verification — reload one model and confirm predictions match
    with open('./exp2_all_features.pkl', 'rb') as _f:
        loaded_model = pickle.load(_f)
    reloaded_preds = loaded_model.predict(X_test_e2)
    original_preds = best_model_e2.predict(X_test_e2)
    assert np.allclose(reloaded_preds, original_preds), 'Mismatch after reload!'
    print('Model reload verification PASSED — predictions match.')
    return


if __name__ == "__main__":
    app.run()
