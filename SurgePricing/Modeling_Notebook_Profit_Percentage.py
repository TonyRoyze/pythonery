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
    # Dynamic Pricing — Comprehensive Modeling Notebook v2

    **Target variable:** `profit_percentage` (% profit over historical cost)

    | # | Experiment | Feature Set | Outlier Treatment |
    |---|-----------|-------------|-------------------|
    | 1 | EDA-selected features | EDA subset | None |
    | 2 | All features | Full set | None |
    | 3 | All features | Full set | Univariate (IQR) |
    | 4 | All features | Full set | Multivariate (Isolation Forest) |
    | 5 | SHAP analysis | Full set | None |
    | 6 | SHAP-selected features | `Number_of_Riders`, `Number_of_Drivers` | None |
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
    #%pip install kagglehub shap catboost xgboost lightgbm pygam -q
    return


@app.cell
def _():
    import os, pathlib, pickle, warnings
    warnings.filterwarnings('ignore')

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import shap
    import joblib
    from IPython.display import display, HTML

    from sklearn.model_selection import train_test_split, GridSearchCV, KFold
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
    from sklearn.inspection import PartialDependenceDisplay

    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    return (
        ColumnTransformer,
        DecisionTreeRegressor,
        ElasticNet,
        GradientBoostingRegressor,
        GridSearchCV,
        IsolationForest,
        KFold,
        LGBMRegressor,
        Lasso,
        LinearRegression,
        OneHotEncoder,
        OrdinalEncoder,
        PartialDependenceDisplay,
        Pipeline,
        RandomForestRegressor,
        Ridge,
        StandardScaler,
        XGBRegressor,
        display,
        joblib,
        make_scorer,
        mean_absolute_error,
        mean_squared_error,
        np,
        pd,
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
    # path = kagglehub.dataset_download('arashnic/dynamic-pricing-dataset')
    # df = pd.read_csv(os.path.join(path, 'dynamic_pricing.csv'))
    df = pd.read_csv('dynamic_pricing.csv')
    print(f'Shape: {df.shape}')
    print(f'Duplicates: {df.duplicated().sum()}')
    df.info()
    return (df,)


@app.cell
def _(df, display):
    missing = df.isna().sum().reset_index()
    missing.columns = ['features', 'missing_count']
    missing['percentage'] = missing['missing_count'] / df.shape[0] * 100
    missing_only = missing[missing['missing_count'] > 0].sort_values('missing_count', ascending=False).reset_index(drop=True)
    display(missing_only.style.background_gradient()) if len(missing_only) > 0 else print('No missing values.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Create Target Variable (`profit_percentage`)

    ```
    adjusted_ride_cost  = Historical_Cost × max(demand_mult, 0.8) × max(supply_mult, 0.8)
    profit_percentage   = (adjusted_ride_cost − Historical_Cost) / Historical_Cost × 100
    ```
    """)
    return


@app.cell
def _(df, np):
    high_demand_percentile, low_demand_percentile = (75, 25)
    high_supply_percentile, low_supply_percentile = (75, 25)
    demand_threshold_high, demand_threshold_low = (1.2, 0.8)
    supply_threshold_high, supply_threshold_low = (0.8, 1.2)
    df['demand_multiplier'] = np.where(df['Number_of_Riders'] > np.percentile(df['Number_of_Riders'], high_demand_percentile), df['Number_of_Riders'] / np.percentile(df['Number_of_Riders'], high_demand_percentile), df['Number_of_Riders'] / np.percentile(df['Number_of_Riders'], low_demand_percentile))
    df['supply_multiplier'] = np.where(df['Number_of_Drivers'] > np.percentile(df['Number_of_Drivers'], low_supply_percentile), np.percentile(df['Number_of_Drivers'], high_supply_percentile) / df['Number_of_Drivers'], np.percentile(df['Number_of_Drivers'], low_supply_percentile) / df['Number_of_Drivers'])
    df['adjusted_ride_cost'] = df['Historical_Cost_of_Ride'] * (np.maximum(df['demand_multiplier'], demand_threshold_low) * np.maximum(df['supply_multiplier'], supply_threshold_high))
    df['profit_percentage'] = (df['adjusted_ride_cost'] - df['Historical_Cost_of_Ride']) / df['Historical_Cost_of_Ride'] * 100
    df_1 = df.drop(['demand_multiplier', 'supply_multiplier', 'adjusted_ride_cost', 'Historical_Cost_of_Ride'], axis=1)
    print('After feature engineering:', df_1.shape)
    df_1.info()
    return (df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. Feature Groups
    """)
    return


@app.cell
def _():
    all_cat_cols = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']
    all_num_cols = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides',
                    'Average_Ratings', 'Expected_Ride_Duration']

    # Full feature set  (Exps 2, 3, 4, 5)
    numerical_features = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides',
                          'Average_Ratings', 'Expected_Ride_Duration']
    ordinal_features   = ['Customer_Loyalty_Status', 'Vehicle_Type']
    nominal_features   = ['Location_Category', 'Time_of_Booking']

    # EDA-selected subset  (Exps 1, (i), (ii))
    eda_numerical = ['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration']
    eda_ordinal   = ['Vehicle_Type']
    eda_nominal   = ['Time_of_Booking']

    # SHAP-selected  (Exp 6)  — top features after SHAP
    shap_numerical = ['Number_of_Riders', 'Number_of_Drivers']
    return (
        all_num_cols,
        eda_nominal,
        eda_numerical,
        eda_ordinal,
        nominal_features,
        numerical_features,
        ordinal_features,
        shap_numerical,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5. Shared Utilities
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
        num_t = Pipeline([('scaler', StandardScaler())])
        nom_t = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
        enc_loy = OrdinalEncoder(categories=[['Regular', 'Silver', 'Gold']])
        enc_veh = OrdinalEncoder(categories=[['Economy', 'Premium']])
        trans = [('num', num_t, num_feats)]
        if 'Customer_Loyalty_Status' in ord_feats: trans.append(('loyl', enc_loy, ['Customer_Loyalty_Status']))
        if 'Vehicle_Type' in ord_feats:            trans.append(('vehi', enc_veh, ['Vehicle_Type']))
        if nom_feats:                              trans.append(('nomi', nom_t,   nom_feats))
        return ColumnTransformer(transformers=trans)

    return (build_preprocessor,)


@app.cell
def _(
    DecisionTreeRegressor,
    ElasticNet,
    GradientBoostingRegressor,
    KFold,
    LGBMRegressor,
    Lasso,
    LinearRegression,
    RandomForestRegressor,
    Ridge,
    XGBRegressor,
    make_scorer,
    mean_absolute_error,
):
    cv         = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    def build_models_config(extended=False):
        cfg = {
            'Linear Regression': {'model': LinearRegression(), 'params': None},
            'Ridge':  {'model': Ridge(random_state=42),
                       'params': {'model__alpha': [0.001,0.01,0.1,1.0,10.0,100.0]}},
            'Lasso':  {'model': Lasso(random_state=42, max_iter=2000),
                       'params': {'model__alpha': [0.0001,0.001,0.01,0.1,1.0,10.0]}},
            'Elastic Net': {'model': ElasticNet(random_state=42, max_iter=2000),
                            'params': {'model__alpha':[0.0001,0.001,0.01,0.1,1.0],
                                       'model__l1_ratio':[0.1,0.3,0.5,0.7,0.9]}},
            'Decision Tree': {'model': DecisionTreeRegressor(random_state=42),
                              'params': {'model__max_depth':[5,10,15,20,None],
                                         'model__min_samples_split':[2,5,10],
                                         'model__min_samples_leaf':[1,2,4]}},
            'Random Forest':  {'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                               'params': {'model__n_estimators':[50,100,200],
                                          'model__max_depth':[10,20,None],
                                          'model__min_samples_split':[2,5,10]}},
            'Gradient Boosting': {'model': GradientBoostingRegressor(random_state=42),
                                  'params': {'model__n_estimators':[50,100,200],
                                             'model__learning_rate':[0.01,0.1,0.2],
                                             'model__max_depth':[3,5,7]}},
        }
        if extended:
            cfg['XGBoost']  = {'model': XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
                               'params': {'model__n_estimators':[100,200,300],
                                          'model__learning_rate':[0.01,0.05,0.1],
                                          'model__max_depth':[3,5,7]}}
            cfg['LightGBM'] = {'model': LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
                               'params': {'model__n_estimators':[100,200,300],
                                          'model__learning_rate':[0.01,0.05,0.1],
                                          'model__num_leaves':[31,63,127]}}
            # cfg['CatBoost'] = {'model': CatBoostRegressor(random_state=42, verbose=0),
            #                    'params': {'model__iterations':[200,500],
            #                               'model__learning_rate':[0.01,0.05,0.1],
            #                               'model__depth':[4,6,8]}}
        return cfg

    return build_models_config, cv, mae_scorer


@app.cell
def _(
    GridSearchCV,
    Pipeline,
    cv,
    display,
    mae_scorer,
    mean_squared_error,
    np,
    pd,
    plt,
    r2_score,
    sns,
):
    def run_experiment(X_train, X_test, y_train, y_test, preprocessor, models_config):
        results = {}
        for name, cfg in models_config.items():
            pipe = Pipeline([('preprocessor', preprocessor), ('model', cfg['model'])])
            if cfg['params']:
                print(f'  Tuning {name}...')
                gs = GridSearchCV(pipe, cfg['params'], cv=cv, scoring=mae_scorer, n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)
                best_pipe = gs.best_estimator_
                print(f'    Best: {gs.best_params_}')
            else:
                best_pipe = pipe.fit(X_train, y_train)

            ytr = best_pipe.predict(X_train); yte = best_pipe.predict(X_test)
            results[name] = {
                'Train_RMSE': np.sqrt(mean_squared_error(y_train, ytr)),
                'Train_R2':   r2_score(y_train, ytr),
                'Test_RMSE':  np.sqrt(mean_squared_error(y_test,  yte)),
                'Test_R2':    r2_score(y_test,  yte),
                'pipeline':   best_pipe,
            }
            print(f'  {name}: Train RMSE={results[name]["Train_RMSE"]:.4f} R2={results[name]["Train_R2"]:.4f}'
                  f' | Test RMSE={results[name]["Test_RMSE"]:.4f} R2={results[name]["Test_R2"]:.4f}')

        cdf       = pd.DataFrame.from_dict(results,'index')[['Train_RMSE','Train_R2','Test_RMSE','Test_R2']]
        best_name = cdf['Test_RMSE'].idxmin()
        print(f'\n>>> Best: {best_name}  Test RMSE={results[best_name]["Test_RMSE"]:.4f}  R2={results[best_name]["Test_R2"]:.4f}')
        display(
            cdf.style
            .background_gradient(cmap='RdYlGn_r', subset=['Train_RMSE','Test_RMSE'])
            .background_gradient(cmap='RdYlGn',   subset=['Train_R2',  'Test_R2'])
            .format({'Train_RMSE':'{:.4f}','Test_RMSE':'{:.4f}','Train_R2':'{:.4f}','Test_R2':'{:.4f}'})
            .set_caption('Model Comparison  (green = better)')
            .set_table_styles([{
                'selector':'caption','props':[('font-size','13px'),('font-weight','bold'),('color','#1e1e2f'),('padding','6px 0')]
            },{'selector':'th','props':[('background-color','#1e1e2f'),('color','#f0f0f0'),('padding','8px 14px'),('text-align','center')]},
             {'selector':'td','props':[('padding','7px 14px'),('text-align','center')]}])
        )
        return results, best_name, results[best_name]['pipeline']


    def plot_predictions(y_test, y_pred, model_name, label='', ylabel='Profit %'):
        residuals = y_test - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=axes[0], color='steelblue')
        lo, hi = min(y_test.min(),y_pred.min()), max(y_test.max(),y_pred.max())
        axes[0].plot([lo,hi],[lo,hi],'r--',lw=2)
        axes[0].set_title(f'{model_name} ({label}): Actual vs Predicted', fontsize=13)
        axes[0].set_xlabel(f'Actual {ylabel}'); axes[0].set_ylabel(f'Predicted {ylabel}')
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=axes[1], color='teal')
        axes[1].axhline(0, color='red', linestyle='--', lw=2)
        axes[1].set_title(f'{model_name} ({label}): Residuals', fontsize=13)
        axes[1].set_xlabel(f'Predicted {ylabel}'); axes[1].set_ylabel('Residuals')
        plt.tight_layout(); plt.show()

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
        n = len(df)
        rows = []
        for col in cols:
            Q1,Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3-Q1; lb,ub = Q1-1.5*IQR, Q3+1.5*IQR
            n_out = ((df[col]<lb)|(df[col]>ub)).sum()
            rows.append({'Feature':col,'Lower':round(lb,2),'Upper':round(ub,2),
                         'Outliers':n_out,'Pct(%)':round(n_out/n*100,2)})
        return pd.DataFrame(rows)


    def remove_univariate_outliers(df, cols):
        df_c = df.copy()
        print(f'Original: {df_c.shape}')
        for col in cols:
            Q1,Q3 = df_c[col].quantile(0.25), df_c[col].quantile(0.75)
            IQR = Q3-Q1; lb,ub = Q1-1.5*IQR, Q3+1.5*IQR
            before = len(df_c)
            df_c = df_c[(df_c[col]>=lb)&(df_c[col]<=ub)]
            print(f'  {col}: removed {before-len(df_c)} rows → {df_c.shape}')
        return df_c


    def remove_multivariate_outliers(df, num_feats, ord_feats, nom_feats,
                                      contamination=0.05, random_state=42):
        """Note: IsolationForest is tree-based; scaling has no effect."""
        df_c = df.copy()
        prep = build_preprocessor(num_feats, ord_feats, nom_feats)
        X_p  = prep.fit_transform(df_c)
        df_c['_anom'] = IsolationForest(contamination=contamination,
                                         random_state=random_state, n_jobs=-1).fit_predict(X_p)
        idx = df_c[df_c['_anom']==-1].index.tolist()
        print(f'Outliers: {len(idx)} ({len(idx)/len(df)*100:.1f}%) → clean: {len(df)-len(idx)}')
        return df.drop(index=idx), idx

    return (
        detect_univariate_outliers,
        remove_multivariate_outliers,
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
    eda_nominal,
    eda_numerical,
    eda_ordinal,
    run_experiment,
    train_test_split,
):
    print('=' * 60, '\nEXPERIMENT 1: EDA Features — No Outlier Removal\n' + '=' * 60)
    X_e1 = df_1[eda_ordinal + eda_nominal + eda_numerical]
    y_e1 = df_1['profit_percentage']
    X_train_e1, X_test_e1, y_train_e1, y_test_e1 = train_test_split(X_e1, y_e1, test_size=0.2, random_state=0)
    prep_e1 = build_preprocessor(eda_numerical, eda_ordinal, eda_nominal)
    results_e1, best_name_e1, best_model_e1 = run_experiment(X_train_e1, X_test_e1, y_train_e1, y_test_e1, prep_e1, build_models_config())
    return X_test_e1, best_model_e1, best_name_e1, y_test_e1


@app.cell
def _(X_test_e1, best_model_e1, best_name_e1, plot_predictions, y_test_e1):
    y_pred_e1 = best_model_e1.predict(X_test_e1)
    plot_predictions(y_test_e1, y_pred_e1, best_name_e1, 'EDA features')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 8. Experiment 2 — All Features (No Outlier Removal)
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
    print('=' * 60, '\nEXPERIMENT 2: All Features — No Outlier Removal\n' + '=' * 60)
    X_e2 = df_1[ordinal_features + nominal_features + numerical_features]
    y_e2 = df_1['profit_percentage']
    X_train_e2, X_test_e2, y_train_e2, y_test_e2 = train_test_split(X_e2, y_e2, test_size=0.2, random_state=0)
    prep_e2 = build_preprocessor(numerical_features, ordinal_features, nominal_features)
    results_e2, best_name_e2, best_model_e2 = run_experiment(X_train_e2, X_test_e2, y_train_e2, y_test_e2, prep_e2, build_models_config())
    return X_test_e2, X_train_e2, best_model_e2, best_name_e2, y_test_e2


@app.cell
def _(X_test_e2, best_model_e2, best_name_e2, plot_predictions, y_test_e2):
    y_pred_e2 = best_model_e2.predict(X_test_e2)
    plot_predictions(y_test_e2, y_pred_e2, best_name_e2, 'All features')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 9. Univariate Outlier Analysis (IQR)
    """)
    return


@app.cell
def _(all_num_cols, detect_univariate_outliers, df_1, display):
    rpt = detect_univariate_outliers(df_1, all_num_cols)
    display(rpt.style.background_gradient(cmap='YlOrRd', subset=['Outliers', 'Pct(%)']))
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
    print('=' * 60, '\nEXPERIMENT 3: All Features — Univariate OR\n' + '=' * 60)
    df_e3 = remove_univariate_outliers(df_1, all_num_cols)
    X_e3 = df_e3[ordinal_features + nominal_features + numerical_features]
    y_e3 = df_e3['profit_percentage']
    X_train_e3, X_test_e3, y_train_e3, y_test_e3 = train_test_split(X_e3, y_e3, test_size=0.2, random_state=0)
    prep_e3 = build_preprocessor(numerical_features, ordinal_features, nominal_features)
    results_e3, best_name_e3, best_model_e3 = run_experiment(X_train_e3, X_test_e3, y_train_e3, y_test_e3, prep_e3, build_models_config())
    return X_test_e3, best_model_e3, best_name_e3, y_test_e3


@app.cell
def _(X_test_e3, best_model_e3, best_name_e3, plot_predictions, y_test_e3):
    y_pred_e3 = best_model_e3.predict(X_test_e3)
    plot_predictions(y_test_e3, y_pred_e3, best_name_e3, 'All + Univariate OR')
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
    eda_nominal,
    eda_numerical,
    eda_ordinal,
    remove_univariate_outliers,
    run_experiment,
    train_test_split,
):
    print('=' * 60, '\nEXPERIMENT (i): EDA Features — Univariate OR\n' + '=' * 60)
    df_ei = remove_univariate_outliers(df_1, eda_numerical + ['profit_percentage'])
    X_ei = df_ei[eda_ordinal + eda_nominal + eda_numerical]
    y_ei = df_ei['profit_percentage']
    X_train_ei, X_test_ei, y_train_ei, y_test_ei = train_test_split(X_ei, y_ei, test_size=0.2, random_state=0)
    prep_ei = build_preprocessor(eda_numerical, eda_ordinal, eda_nominal)
    results_ei, best_name_ei, best_model_ei = run_experiment(X_train_ei, X_test_ei, y_train_ei, y_test_ei, prep_ei, build_models_config())
    return X_test_ei, best_model_ei, best_name_ei, y_test_ei


@app.cell
def _(X_test_ei, best_model_ei, best_name_ei, plot_predictions, y_test_ei):
    y_pred_ei = best_model_ei.predict(X_test_ei)
    plot_predictions(y_test_ei, y_pred_ei, best_name_ei, 'EDA + Univariate OR')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 10. Multivariate Outlier Detection — Isolation Forest

    ## 10a. Experiment 4 — All Features + Multivariate Outlier Removal

    > Isolation Forest is tree-based; data scaling has no effect.
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
    remove_multivariate_outliers,
    run_experiment,
    train_test_split,
):
    print('=' * 60, '\nEXPERIMENT 4: All Features — Multivariate OR\n' + '=' * 60)
    df_e4, idx_e4 = remove_multivariate_outliers(df_1, numerical_features, ordinal_features, nominal_features)
    X_e4 = df_e4[ordinal_features + nominal_features + numerical_features]
    y_e4 = df_e4['profit_percentage']
    X_train_e4, X_test_e4, y_train_e4, y_test_e4 = train_test_split(X_e4, y_e4, test_size=0.2, random_state=0)
    prep_e4 = build_preprocessor(numerical_features, ordinal_features, nominal_features)
    results_e4, best_name_e4, best_model_e4 = run_experiment(X_train_e4, X_test_e4, y_train_e4, y_test_e4, prep_e4, build_models_config())
    return X_test_e4, best_model_e4, best_name_e4, y_test_e4


@app.cell
def _(X_test_e4, best_model_e4, best_name_e4, plot_predictions, y_test_e4):
    y_pred_e4 = best_model_e4.predict(X_test_e4)
    plot_predictions(y_test_e4, y_pred_e4, best_name_e4, 'All + Multivariate OR')
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
    eda_nominal,
    eda_numerical,
    eda_ordinal,
    remove_multivariate_outliers,
    run_experiment,
    train_test_split,
):
    print('=' * 60, '\nEXPERIMENT (ii): EDA Features — Multivariate OR\n' + '=' * 60)
    df_eii, idx_eii = remove_multivariate_outliers(df_1, eda_numerical, eda_ordinal, eda_nominal)
    X_eii = df_eii[eda_ordinal + eda_nominal + eda_numerical]
    y_eii = df_eii['profit_percentage']
    X_train_eii, X_test_eii, y_train_eii, y_test_eii = train_test_split(X_eii, y_eii, test_size=0.2, random_state=0)
    prep_eii = build_preprocessor(eda_numerical, eda_ordinal, eda_nominal)
    results_eii, best_name_eii, best_model_eii = run_experiment(X_train_eii, X_test_eii, y_train_eii, y_test_eii, prep_eii, build_models_config())
    return X_test_eii, best_model_eii, best_name_eii, y_test_eii


@app.cell
def _(X_test_eii, best_model_eii, best_name_eii, plot_predictions, y_test_eii):
    y_pred_eii = best_model_eii.predict(X_test_eii)
    plot_predictions(y_test_eii, y_pred_eii, best_name_eii, 'EDA + Multivariate OR')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 11. SHAP Feature Importance Analysis

    Using the best model from **Experiment 2** (All Features, No Outlier Removal).
    """)
    return


@app.cell
def _(best_model_e2, best_name_e2):
    prep_shap  = best_model_e2.named_steps['preprocessor']
    model_shap = best_model_e2.named_steps['model']
    feat_names = prep_shap.get_feature_names_out()
    print(f'Best model: {best_name_e2}  |  Features after preprocessing: {len(feat_names)}')
    return feat_names, model_shap, prep_shap


@app.cell
def _(display, feat_names, model_shap, pd):
    if hasattr(model_shap, 'feature_importances_'):
        imp_df = (pd.DataFrame({'Feature':feat_names,'Importance':model_shap.feature_importances_})
                  .sort_values('Importance', ascending=False).reset_index(drop=True))
        display(imp_df.head(15).style.background_gradient(cmap='Blues', subset=['Importance']))
    return


@app.cell
def _(X_train_e2, feat_names, model_shap, pd, prep_shap, shap):
    X_tr_prep = prep_shap.transform(X_train_e2)
    if hasattr(X_tr_prep, 'toarray'): X_tr_prep = X_tr_prep.toarray()
    X_tr_prep_df = pd.DataFrame(X_tr_prep, columns=feat_names)

    explainer   = shap.TreeExplainer(model_shap)
    shap_values = explainer.shap_values(X_tr_prep_df)
    print('SHAP values computed.')
    return X_tr_prep_df, shap_values


@app.cell
def _(X_tr_prep_df, feat_names, plt, shap, shap_values):
    plt.figure(figsize=(10,8))
    shap.summary_plot(shap_values, X_tr_prep_df, plot_type='dot', feature_names=feat_names, show=False)
    plt.title('SHAP Summary — Feature Impact on Profit %')
    plt.tight_layout(); plt.show()
    return


@app.cell
def _(X_tr_prep_df, feat_names, np, plt, shap, shap_values):
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argmax(mean_abs)
    top_name = feat_names[top_idx]
    print(f'Most important feature: {top_name}')
    plt.figure(figsize=(10,6))
    shap.dependence_plot(top_name, shap_values, X_tr_prep_df, interaction_index='auto', show=False)
    plt.title(f'SHAP Dependence — {top_name}'); plt.tight_layout(); plt.show()
    return


@app.cell
def _(
    X_train_e2,
    display,
    feat_names,
    nominal_features,
    np,
    numerical_features,
    ordinal_features,
    pd,
    plt,
    shap_values,
    sns,
):
    shap_df = pd.DataFrame(shap_values, columns=feat_names, index=X_train_e2.index)
    shap_var = {}
    for var in nominal_features:
        cols = [c for c in feat_names if c.startswith(f'nomi__{var}')]
        if cols: shap_var[var] = np.abs(shap_df[cols]).mean().sum()
    for var in ordinal_features:
        cols = [c for c in feat_names if var in c]
        if cols: shap_var[var] = np.abs(shap_df[cols]).mean().sum()
    for var in numerical_features:
        cols = [c for c in feat_names if f'num__{var}' in c or c==var]
        if cols: shap_var[var] = np.abs(shap_df[cols]).mean().sum()

    shap_var_df = (pd.DataFrame({'Variable':list(shap_var),'Mean_Abs_SHAP':list(shap_var.values())})
                   .sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True))
    display(shap_var_df.style.background_gradient(cmap='Greens', subset=['Mean_Abs_SHAP']))

    plt.figure(figsize=(10,6))
    sns.barplot(data=shap_var_df, x='Mean_Abs_SHAP', y='Variable', palette='viridis')
    plt.title('Variable-wise SHAP Importance'); plt.xlabel('Mean |SHAP|'); plt.tight_layout(); plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 12. Experiment 6 — SHAP-Selected Features (Extended Models)

    Top SHAP features: `Number_of_Riders`, `Number_of_Drivers`.

    Includes XGBoost, LightGBM, and CatBoost.
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    StandardScaler,
    build_models_config,
    df_1,
    run_experiment,
    shap_numerical,
    train_test_split,
):
    print('=' * 60, '\nEXPERIMENT 6: SHAP-Selected Features\n' + '=' * 60)
    X_e6 = df_1[shap_numerical]
    y_e6 = df_1['profit_percentage']
    X_train_e6, X_test_e6, y_train_e6, y_test_e6 = train_test_split(X_e6, y_e6, test_size=0.2, random_state=0)
    prep_e6 = ColumnTransformer([('num', StandardScaler(), shap_numerical)])
    results_e6, best_name_e6, best_model_e6 = run_experiment(X_train_e6, X_test_e6, y_train_e6, y_test_e6, prep_e6, build_models_config(extended=True))
    return X_test_e6, X_train_e6, best_model_e6, best_name_e6, y_test_e6


@app.cell
def _(X_test_e6, best_model_e6, best_name_e6, plot_predictions, y_test_e6):
    y_pred_e6 = best_model_e6.predict(X_test_e6)
    plot_predictions(y_test_e6, y_pred_e6, best_name_e6, 'SHAP features')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12a. Partial Dependence Plots (Best Model — Exp 6)
    """)
    return


@app.cell
def _(
    PartialDependenceDisplay,
    X_train_e6,
    best_model_e6,
    plt,
    shap_numerical,
):
    X_train_e6_copy = X_train_e6.copy().astype(float)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title('Partial Dependence Plots — SHAP-Selected Features')
    PartialDependenceDisplay.from_estimator(best_model_e6, X_train_e6_copy,
        features=shap_numerical, ax=ax)
    plt.tight_layout(); plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 13. Export Best Models

    Each experiment's best pipeline is saved to `exported_models_v2/` as a pickle file.
    """)
    return


@app.cell
def _(EXPORT_DIR, best_model_e6, joblib):
    models_to_export = {
        # 'exp2.1_eda_features':         best_model_e1,
        # 'exp2.2_all_features':         best_model_e2,
        # 'exp2.3_all_univariate_or':    best_model_e3,
        # 'exp2.4_all_multivariate_or':  best_model_e4,
        'exp2.6_shap_features':        best_model_e6,
        # 'exp2.1.1.i_eda_univariate_or':    best_model_e1_i,
        # 'exp2.1.1.ii_eda_multivariate_or': best_model_e1_ii,
    }

    for key, pipe in models_to_export.items():
        out = f'./{key}.pkl'
        joblib.dump(pipe, out)
        print(f'Saved: {out}')

    print(f'\n{len(models_to_export)} models exported to {EXPORT_DIR.resolve()}')
    return


@app.cell
def _(X_test_e2, best_model_e2, joblib, np):
    # Verify reload
    loaded = joblib.load('./exp2_all_features.pkl')
    assert np.allclose(loaded.predict(X_test_e2), best_model_e2.predict(X_test_e2))
    print('Reload verification PASSED.')
    return


if __name__ == "__main__":
    app.run()
