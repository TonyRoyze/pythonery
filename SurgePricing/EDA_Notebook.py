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
    #**Dynamic Pricing**
    *Maximize revenue and profitability by dynamic pricing*
    :
    [link text](https://www.kaggle.com/datasets/arashnic/dynamic-pricing-dataset/code?datasetId=4365344&sortBy=voteCount&language=Python&outputs=Visualization)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #**Description of the problem**

    *A ride-sharing company currently determines ride fares solely based on ride duration. However, this approach does not account for real-time market conditions such as demand, supply availability, and customer behavior, which can significantly impact optimal pricing.*

    *To improve its pricing strategy, the company plans to implement a dynamic pricing model using historical ride data and data-driven techniques.*

    *Dynamic pricing is widely used by most ride-sharing platforms (about 65% of them)*:
    [link text](https://www.winsavvy.com/dynamic-pricing-adoption-rates-by-industry/?utm_source=chatgpt.com)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #**Objectives**

    *1.To understand how factors such as demand, number of available drivers, and customer characteristics affect ride prices.*

    *2.To build a model that predicts ride fares using past ride data and current market conditions.*

    *3.To apply a dynamic pricing approach that changes fares based on real-time conditions to improve revenue and customer satisfaction.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Import Python Libraries**
    """)
    return


@app.cell
def _():
    #%pip install kagglehub numpy pandas matplotlib seaborn statsmodels diptest prince scikit-learn
    return


@app.cell
def _():
    import kagglehub
    import pandas as pd
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    #Split
    from sklearn.model_selection import train_test_split
    from IPython.display import display
    import diptest

    #FAMD
    import prince


    #Clustering
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import silhouette_samples
    import matplotlib.cm as cm

    #Mutaul Information
    from sklearn.feature_selection import mutual_info_regression

    #Skewness
    from scipy.stats import skew

    #Scaling
    from sklearn.preprocessing import StandardScaler


    #GVIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    #
    from scipy.spatial.distance import mahalanobis
    from numpy.linalg import inv

    return (
        KMeans,
        display,
        inv,
        kagglehub,
        mahalanobis,
        mutual_info_regression,
        np,
        os,
        pd,
        plt,
        prince,
        silhouette_samples,
        silhouette_score,
        sns,
        train_test_split,
        variance_inflation_factor,
    )


@app.cell
def _():
    #shapiro wilk test
    from scipy.stats import shapiro

    #Anderson-Darling test
    from scipy.stats import anderson

    #Spearman Correlation
    from scipy.stats import spearmanr

    #Levene’s test
    from scipy.stats import levene

    #Kruskal-Wallis H-test
    from scipy.stats import kruskal

    return anderson, kruskal, levene, shapiro, spearmanr


@app.cell
def _(sns):
    sns.set_theme()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Load Dataset**
    """)
    return


@app.cell
def _(kagglehub):
    path = kagglehub.dataset_download('arashnic/dynamic-pricing-dataset')
    return (path,)


@app.cell
def _(os, path, pd):
    df_org = pd.read_csv(os.path.join(path, "dynamic_pricing.csv"))
    return (df_org,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Original Dataset**
    """)
    return


@app.cell
def _(df_org):
    df_org.head()
    return


@app.cell
def _(df_org):
    df_org.info()
    return


@app.cell
def _(df_org):
    #check shape of the data
    print(f'The original Dataset has {df_org.shape[0]} rows and {df_org.shape[1]} columns.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Data Preprocessing**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Check and Handle Duplicates**
    """)
    return


@app.cell
def _(df_org):
    #check duplicates
    df_org.duplicated().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Check and Handle Missing values**
    """)
    return


@app.cell
def _(df_org, display):
    # Check missing values in original dataset
    missing = df_org.isna().sum().reset_index()
    missing.columns = ['features', 'missing_count']

    # Calculate the percentage of missing values in each column
    missing['percentage'] = missing['missing_count'] / df_org.shape[0] * 100

    # Filter columns with missing values
    missing_only = missing[missing['missing_count'] > 0].sort_values(by='missing_count', ascending=False).reset_index(drop=True)

    display(missing_only.style.background_gradient())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Create Target Variable**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Create Target Variable-Adjusted Ride Cost**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calculate 'demand_multiplier' for each record based on rider demand:
    Define high demand as values above the 75th percentile and low demand as values below the 25th percentile.
    If a record has high demand, the multiplier is riders / 75th percentile ( > 1 ).
    Otherwise, the multiplier is riders / 25th percentile ( ≤ 1 ).
    This helps scale demand relative to typical low and high demand levels in the dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Captures relative demand:

    Not all locations or times have the same number of riders.

    This multiplier turns raw Number_of_Riders into a value that compares demand relative to typical high and low levels.

    Example: A location with 120 riders might be “high demand” compared to the 75th percentile, giving a multiplier >1.
    """)
    return


@app.cell
def _(df_org, np):
    # Calculate demand_multiplier based on percentile for high and low demand
    high_demand_percentile = 75
    low_demand_percentile = 25

    df_org['demand_multiplier'] = np.where(df_org['Number_of_Riders'] > np.percentile(df_org['Number_of_Riders'], high_demand_percentile),
                                         df_org['Number_of_Riders'] / np.percentile(df_org['Number_of_Riders'], high_demand_percentile),
                                         df_org['Number_of_Riders'] / np.percentile(df_org['Number_of_Riders'], low_demand_percentile))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calculate 'supply_multiplier' for each record based on rider demand:
    Define high supply as values above the 75th percentile and low supply as values below the 25th percentile.
    If a record has high supply, the multiplier is drivers / 75th percentile ( > 1 ).
    Otherwise, the multiplier is drivers / 25th percentile ( ≤ 1 ).
    This helps scale supply relative to typical low and high supply levels in the dataset.
    """)
    return


@app.cell
def _(df_org, np):
    # Calculate supply_multiplier based on percentile for high and low supply
    high_supply_percentile = 75
    low_supply_percentile = 25

    df_org['supply_multiplier'] = np.where(df_org['Number_of_Drivers'] > np.percentile(df_org['Number_of_Drivers'], low_supply_percentile),
                                         np.percentile(df_org['Number_of_Drivers'], high_supply_percentile) / df_org['Number_of_Drivers'],
                                         np.percentile(df_org['Number_of_Drivers'], low_supply_percentile) / df_org['Number_of_Drivers'])
    return


@app.cell
def _():
    # Define price adjustment factors for high and low demand
    demand_threshold_high = 1.2  # Higher demand threshold
    demand_threshold_low = 0.8  # Lower demand threshold
    return demand_threshold_high, demand_threshold_low


@app.cell
def _(demand_threshold_high, demand_threshold_low, df_org, plt):
    plt.hist(df_org['demand_multiplier'], alpha=0.7, edgecolor='k')
    plt.axvline(demand_threshold_high, color='red', linestyle='--', label='High Threshold')
    plt.axvline(demand_threshold_low, color='blue', linestyle='--', label='Low Threshold')
    plt.legend()
    plt.title("Distribution of Demand Multiplier")
    plt.show()
    return


@app.cell
def _():
    # Define price adjustment factors for high and low supply
    supply_threshold_high = 0.8  # Higher supply threshold
    supply_threshold_low = 1.2  # Lower supply threshold
    return supply_threshold_high, supply_threshold_low


@app.cell
def _(df_org, plt, supply_threshold_high, supply_threshold_low):
    plt.hist(df_org['supply_multiplier'], alpha=0.7, edgecolor='k')
    plt.axvline(supply_threshold_high , color='red', linestyle='--', label='High Threshold')
    plt.axvline(supply_threshold_low, color='blue', linestyle='--', label='Low Threshold')
    plt.legend()
    plt.title("Distribution of Supply Multiplier")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Calculate adjusted_ride_cost**
    """)
    return


@app.cell
def _(demand_threshold_low, df_org, np, supply_threshold_high):
    # Calculate adjusted_ride_cost for dynamic pricing
    df_org['adjusted_ride_cost'] = df_org['Historical_Cost_of_Ride'] * (
        np.maximum(df_org['demand_multiplier'], demand_threshold_low) *
        np.maximum(df_org['supply_multiplier'], supply_threshold_high)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3 variables added to the data frame(demand_multiplier,supply_multiplier,adjusted_ride_cost)
    """)
    return


@app.cell
def _(df_org):
    #dropping unwanted columns
    df_org_1 = df_org.drop(['demand_multiplier', 'supply_multiplier'], axis=1)
    return (df_org_1,)


@app.cell
def _(df_org_1):
    df_org_1.head()
    return


@app.cell
def _(df_org_1):
    #dropping the historic cost coloumn, since we got adjusted cost by using historic cost
    #dropping unwanted columns
    df_org_2 = df_org_1.drop(['Historical_Cost_of_Ride'], axis=1)
    return (df_org_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Data frame info after preprocessing and target variable creation**
    """)
    return


@app.cell
def _(df_org_2):
    df_org_2.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Numeric and Categorical columns**
    """)
    return


@app.cell
def _(df_org_2):
    # Numeric columns
    numeric_cols = df_org_2.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df_org_2.select_dtypes(include=['object']).columns.tolist()
    # Categorical columns
    print('Numeric columns:', numeric_cols)
    print('Categorical columns:', categorical_cols)
    return categorical_cols, numeric_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #**Training And Testing Data Split**
    """)
    return


@app.cell
def _(df_org_2):
    #separating features and target for use in modelling
    X = df_org_2.drop(['adjusted_ride_cost'], axis=1)
    y = df_org_2['adjusted_ride_cost']
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    train-80%,test-20%,random state=0
    """)
    return


@app.cell
def _(X, train_test_split, y):
    # Splitting dataset to train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=0)
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_train, pd, y_train):
    #creating train dataset
    df_train=pd.concat([X_train, y_train], axis=1)
    return (df_train,)


@app.cell
def _(X_test, pd, y_test):
    #creating test dataset
    df_test=pd.concat([X_test, y_test], axis=1)
    return (df_test,)


@app.cell
def _(df_test, df_train):
    #check shape of data
    print(f'The training dataset has {df_train.shape[0]} rows and {df_train.shape[1]} columns.')
    print(f'The testing dataset has {df_test.shape[0]} rows and {df_test.shape[1]} columns.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Train data**
    """)
    return


@app.cell
def _(df_train):
    df_train.info()
    return


@app.cell
def _(df_train):
    df_train.head()
    return


@app.cell
def _(df_train):
    df_train.describe()
    return


@app.cell
def _(df_train):
    df_train.describe(include='O')
    return


@app.cell
def _(categorical_cols, df_train):
    print('Levels for each categorical variable:\n')
    for _col in categorical_cols:
        print(f'Column: {_col}')
        print(df_train[_col].value_counts())
        print('\n' + '-' * 30 + '\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Factor Analysis for Mixed Data**
    """)
    return


@app.cell
def _(df_train, np, plt, prince, sns):
    famd_train = df_train.drop(columns=['adjusted_ride_cost','Number_of_Riders','Number_of_Drivers'])
    #In a dynamic pricing context, you typically treat Supply and Demand as Interaction Terms or primary inputs. You don't want them buried or "shuffled" inside a Principal Component where their individual impact becomes harder to interpret.#

    famd = prince.FAMD(
        n_components=famd_train.shape[1],
        n_iter=3,
        copy=True,
        check_input=True,
        random_state=42,
        engine="sklearn",
        handle_unknown="error"
    )

    famd = famd.fit(famd_train)

    eigenvalues = famd.eigenvalues_  # λ values
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)  # fraction of variance explained
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)


    plt.figure(figsize=(12, 5))

    # Scree plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
    plt.axhline(y=1, color='red', linestyle='--', label='Kaiser Criterion (λ = 1)')
    plt.title("Scree Plot (FAMD)")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.grid(True)

    # Cumulative variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_explained_variance) + 1),
             cumulative_explained_variance, marker='o')
    plt.axhline(y=0.90, color='green', linestyle='--', label='90% Variance')
    plt.axhline(y=0.95, color='orange', linestyle='--', label='95% Variance')
    plt.title("Cumulative Explained Variance (FAMD)")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    n_kaiser = np.sum(eigenvalues > 1)
    n_90 = np.argmax(cumulative_explained_variance >= 0.90) + 1
    n_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1

    print("===================================")
    print(f"Original features      : {famd_train.shape[1]}")
    print(f"Kaiser criterion (λ>1) : {n_kaiser} components")
    print(f"90% explained variance : {n_90} components")
    print(f"95% explained variance : {n_95} components")
    print("===================================")

    #Score plot

    # 1. Transform the data to get the FAMD coordinates/dimensions
    famd_scores = famd.transform(famd_train)

    # 2. Plotting the Score Plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=famd_scores.iloc[:, 0],
        y=famd_scores.iloc[:, 1],
        alpha=0.5,
        color='steelblue'  # Single color as requested
    )

    # Labeling with the correct "Dimension" terminology
    plt.title('FAMD Score Plot (Factor 1 vs Factor 2)')
    plt.xlabel(f'Factor 1 ({explained_variance_ratio[0]*100:.2f}%)')
    plt.ylabel(f'Factor 2 ({explained_variance_ratio[1]*100:.2f}%)')

    # Adding the origin lines for reference
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()
    return famd, famd_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *If you want a highly accurate model that captures almost all the information from your categorical and numerical variables, you should choose 7 components.*

    *you do not have distinct, separated clusters.While you don't have clusters, you do have isolated points.Notice the dots sitting far out at the edges (e.g., Dimension 2 near 12.5 or Dimension 1 near -7.5).
    These are the multivariate outliers
    """)
    return


@app.cell
def _(KMeans, famd, famd_train, np, plt, silhouette_samples, silhouette_score):
    famd_components = famd.row_coordinates(famd_train)  # transformed dataset
    silhouette_scores = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(famd_components)
        score = silhouette_score(famd_components, cluster_labels)
        silhouette_scores.append(score)
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, silhouette_scores, marker='o')
    plt.scatter(K_range, silhouette_scores, color='red', zorder=5)
    # Plot silhouette scores
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')  # Add scatter plot for individual scores
    plt.grid(True)
    plt.show()
    K_values = range(2, 11)
    famd_components = famd.row_coordinates(famd_train)
    plt.figure(figsize=(15, 25))
    for _i, k in enumerate(K_values, 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(famd_components)  # k = 2 to 10
        sil_score = silhouette_score(famd_components, cluster_labels)  # transformed dataset
        sil_samples = silhouette_samples(famd_components, cluster_labels)
        plt.subplot(len(K_values), 2, 2 * _i - 1)
        plt.scatter(famd_components[0], famd_components[1], c=cluster_labels, cmap='Set2', s=50)
        plt.title(f'K-Means Clusters (k={k})')
        plt.xlabel('FAMD Component 1')
        plt.ylabel('FAMD Component 2')
        plt.grid(True)
        plt.subplot(len(K_values), 2, 2 * _i)  # Silhouette score
        y_lower = 10
        for j in range(k):
            cluster_sil_values = sil_samples[cluster_labels == j]
            cluster_sil_values.sort()  # Scatter plot of clusters
            size_cluster = cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster
            color = plt.cm.Set2(j / k)
            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil_values, facecolor=color, edgecolor=color, alpha=0.7)
            plt.text(-0.05, y_lower + 0.5 * size_cluster, str(j))
            y_lower = y_upper + 10
        plt.axvline(x=sil_score, color='red', linestyle='--')
        plt.title(f'Silhouette Plot (k={k}, score={sil_score:.2f})')  # Silhouette plot
        plt.xlabel('Silhouette coefficient')
        plt.ylabel('Cluster')
        plt.xlim([-0.1, 1])
        plt.ylim([0, len(famd_components) + (k + 1) * 10])
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *You do NOT have natural clusters in this data.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Mutual Information for all Features**
    """)
    return


@app.cell
def _(df_train, mutual_info_regression, pd, plt, sns):
    X_1 = df_train.drop(columns=['adjusted_ride_cost'])
    y_1 = df_train['adjusted_ride_cost']
    excluded_cols = []
    X_encoded = X_1.copy()
    X_encoded = X_encoded.drop(columns=excluded_cols, errors='ignore')
    for _col in X_encoded.select_dtypes(include='object').columns:
        X_encoded[_col] = X_encoded[_col].astype('category').cat.codes
    mi = mutual_info_regression(X_encoded, y_1, discrete_features='auto', random_state=42)
    mi_df = pd.DataFrame({'Feature': X_encoded.columns, 'Mutual_Information': mi})
    mi_df = mi_df.sort_values(by='Mutual_Information', ascending=False)
    print(mi_df)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Mutual_Information', y='Feature', data=mi_df, palette='viridis')
    plt.title('Mutual Information of Features vs adjusted_ride_cost')
    plt.xlabel('Mutual Information')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Exploratory Data Analysis**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Univariate Analysis**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Investigate Response Variable: adjusted_ride_cost**
    """)
    return


@app.cell
def _(df_train, plt, sns):
    # Histogram with KDE for 'adjusted_ride_cost'
    plt.figure(figsize=(10, 6))
    sns.histplot(df_train['adjusted_ride_cost'], kde=True, bins=30)
    plt.title('Distribution of Adjusted Ride Cost')
    plt.xlabel('Adjusted Ride Cost')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(df_train, plt, sns):
    # Box plot for 'adjusted_ride_cost'
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df_train['adjusted_ride_cost'])
    plt.title('Box Plot of Adjusted Ride Cost')
    plt.ylabel('Adjusted Ride Cost')
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(df_train, display):
    # Summary statistics for the response variable
    premium_summary =df_train['adjusted_ride_cost'].describe()
    display(premium_summary)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Normality Assessment of Adjusted Ride Cost**
    """)
    return


@app.cell
def _(df_train, plt):
    import scipy.stats as stats

    stats.probplot(df_train['adjusted_ride_cost'], dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    H₀:The data follows a normal distribution.

    H₁:The data does not follow a normal distribution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #####**Shapiro-Wilk test**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Interpretation:

    p > 0.05 → Data is normal

    p < 0.05 → Data is NOT normal
    """)
    return


@app.cell
def _(df_train, shapiro):
    _stat, _p = shapiro(df_train['adjusted_ride_cost'])
    print('Statistic:', _stat)
    print('p-value:', _p)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### **Anderson-Darling test**
    """)
    return


@app.cell
def _(anderson, df_train):
    result = anderson(df_train['adjusted_ride_cost'])
    print("Statistic:", result.statistic)
    print("Critical Values:", result.critical_values)
    print("Significance Levels:", result.significance_level)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Both the Shapiro–Wilk and Anderson–Darling tests consistently indicate that the adjusted_ride_cost variable is not normally distributed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Distribution of Numerical Variables**
    """)
    return


@app.cell
def _(plt, sns):
    def plot_hist_box(data, columns, figsize=(14, 4)):
        n = len(columns)
        plt.figure(figsize=(figsize[0], figsize[1] * n))
        for _i, _col in enumerate(columns):
            plt.subplot(n, 2, 2 * _i + 1)
            sns.histplot(data[_col].dropna(), bins=20, kde=True)
            plt.title(f'Distribution of {_col}')  # Histogram + KDE
            plt.xlabel(_col)
            plt.ylabel('Count')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.subplot(n, 2, 2 * _i + 2)
            sns.boxplot(y=data[_col])
            plt.title(f'Distribution of {_col}')
            plt.ylabel(_col)
            plt.grid(axis='y', linestyle='--', alpha=0.7)  # Boxplot
        plt.tight_layout()
        plt.show()

    return (plot_hist_box,)


@app.cell
def _(df_train, plot_hist_box):
    plot_hist_box(df_train, [_col for _col in df_train.select_dtypes(include=['int64', 'float64']).columns if _col != 'profit_percentage'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Distribution of Categorical Variable**
    """)
    return


@app.cell
def _(plt, sns):
    def plot_categorical_distributions(df, categorical_cols):
        for _col in categorical_cols:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            sns.countplot(data=df, x=_col, ax=axes[0], palette='viridis', hue=_col, legend=False)
            axes[0].set_title(f'Count Plot of {_col}')  # Count Plot
            axes[0].set_xlabel(_col)
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
            counts = df[_col].value_counts()
            labels = counts.index
            sizes = counts.values
            axes[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(labels)))  # Pie Chart
            axes[1].set_title(f'Pie Chart of {_col}')
            axes[1].axis('equal')
            plt.tight_layout()
            plt.show()  # Equal aspect ratio ensures that pie is drawn as a circle.

    return (plot_categorical_distributions,)


@app.cell
def _(categorical_cols, df_train, plot_categorical_distributions):
    # Call the function to visualize categorical distributions
    plot_categorical_distributions(df_train, categorical_cols)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Bivariate Analysis**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *Relationship of Predictor Variables with the response- Adjusted Cost*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To understand the relationship between the predictor variables and the `adjusted_ride_cost` response variable, I'll generate the following plots:

    1.  **Bivariate Numerical Plots:** Scatter plots for each numerical predictor against `adjusted_ride_cost`.
    2.  **Bivariate Categorical Plots:** Box plots for each categorical predictor against `adjusted_ride_cost`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## bivariate_numerical_plots
    """)
    return


@app.cell
def _(df_train, numeric_cols, plt, sns):
    numerical_predictors = [_col for _col in numeric_cols if _col not in ['adjusted_ride_cost', 'profit_percentage']]
    for _col in numerical_predictors:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_train, x=_col, y='adjusted_ride_cost')
        plt.title(f'Scatter Plot of  Adjusted Ride Cost vs.{_col} ')
        plt.xlabel(_col)
        plt.ylabel('Adjusted Ride Cost')
        plt.grid(True)
        plt.show()
    return (numerical_predictors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Correlation Analysis of Numerical Predictors with Adjusted Ride Cost**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Spearman Rank Correlation Hypotheses

    **Null Hypothesis (H₀):**

    There is no monotonic relationship between the predictor and the response variable

    **Alternative Hypothesis (H₁):**

    There is a monotonic relationship (either increasing or decreasing) between the predictor and the response variable.
    """)
    return


@app.cell
def _(df_train, display, numerical_predictors, pd, spearmanr):
    # List to store results
    spearman_results = []
    for _col in numerical_predictors:
        rho, p_value = spearmanr(df_train[_col], df_train['adjusted_ride_cost'])
        abs_rho = abs(rho)
        if abs_rho < 0.2:
            strength = 'Very weak'  # Determine strength
        elif abs_rho < 0.4:
            strength = 'Weak'
        elif abs_rho < 0.6:
            strength = 'Moderate'
        elif abs_rho < 0.8:
            strength = 'Strong'
        else:
            strength = 'Very strong'
        significance = 'Significant' if p_value < 0.05 else 'Not significant'
        direction = 'Positive' if rho > 0 else 'Negative'
        interpretation = f'{strength} {direction}, {significance}'
        spearman_results.append({'Predictor': _col, 'Spearman_rho': rho, 'p_value': p_value, 'Interpretation': interpretation})
    spearman_df = pd.DataFrame(spearman_results)  # Determine significance
    spearman_df['abs_rho'] = spearman_df['Spearman_rho'].abs()
    spearman_df = spearman_df.sort_values(by='abs_rho', ascending=False)
    # Convert to DataFrame
    # Sort by absolute correlation to see strongest relationships
    # Display final table
    display(spearman_df[['Predictor', 'Spearman_rho', 'p_value', 'Interpretation']])  # Determine direction  # Combine interpretation  # Store results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | Predictor        | Interpretation                                                                            |
    | ---------------- | ----------------------------------------------------------------------------------------- |
    | distance_km      | Moderate positive and significant → longer distances increase ride cost                   |
    | ride_duration    | Weak positive and significant → longer rides slightly increase cost                       |
    | Number_of_Riders | Very weak negative, not significant → number of riders does not affect cost               |
    | vehicle_age      | Weak negative, not significant → older vehicles slightly reduce cost, but not significant |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## bivariate_categorical_plots
    """)
    return


@app.cell
def _(categorical_cols, df_train, plt, sns):
    for _col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_train, x=_col, y='adjusted_ride_cost', palette='viridis')
        plt.title(f'Box Plot of Adjusted Ride Cost vs. {_col}')
        plt.xlabel(_col)
        plt.ylabel('Adjusted Ride Cost')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Multivariate Analysis**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Numeric vs Numeric: Correlation Heatmap + Pairplots + VIF*
    """)
    return


@app.cell
def _(df_train, pd, plt, sns, variance_inflation_factor):
    import statsmodels.api as sm
    numeric_cols_1 = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration']
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_train[numeric_cols_1].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.show()
    sns.pairplot(df_train[numeric_cols_1])
    plt.show()
    X_2 = df_train[numeric_cols_1]
    X_2 = sm.add_constant(X_2)
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X_2.columns
    vif_data['VIF'] = [variance_inflation_factor(X_2.values, _i) for _i in range(X_2.shape[1])]
    print(vif_data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Multicollinearity analysis shows a healthy dataset: the moderate correlation of 0.62 between Number_of_Riders and Number_of_Drivers is a realistic reflection of market supply and demand. Because the VIF values for these features are low (approx. 1.62), they are mathematically safe to use together in your dynamic pricing model.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *GVIF (Generalized VIF)*
    """)
    return


@app.cell
def _(df_train, np, pd, variance_inflation_factor):
    numeric_cols_2 = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration']
    categorical_cols_1 = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']
    X_3 = df_train[numeric_cols_2 + categorical_cols_1].copy()
    X_3 = pd.get_dummies(X_3, drop_first=True)
    X_3 = X_3.apply(pd.to_numeric, errors='coerce').astype(float)
    X_3 = X_3.loc[:, X_3.std() > 1e-05]
    X_3 = X_3.replace([np.inf, -np.inf], np.nan).dropna()
    vif_values = []
    for _i in range(X_3.shape[1]):
        try:
            vif_values.append(variance_inflation_factor(X_3.values, _i))
        except:
            vif_values.append(np.inf)
    vif_df = pd.DataFrame({'Feature': X_3.columns, 'VIF': vif_values})
    gvif_results = []
    for var in numeric_cols_2 + categorical_cols_1:
        related_cols = [c for c in vif_df['Feature'] if c.startswith(var)]
        df_var = len(related_cols)
        if df_var == 0:
            continue
        if df_var == 1:
            gvif = vif_df.loc[vif_df['Feature'] == related_cols[0], 'VIF'].values[0]
        else:
            gvif = vif_df.loc[vif_df['Feature'].isin(related_cols), 'VIF'].prod()
        gvif_adj = gvif ** (1 / (2 * df_var))
        gvif_results.append({'Variable': var, 'GVIF': gvif, 'Df': df_var, 'GVIF^(1/(2*Df))': gvif_adj})
    gvif_df = pd.DataFrame(gvif_results)
    print(gvif_df)
    return (numeric_cols_2,)


@app.cell
def _(plt, sns, temp_df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=temp_df, x='Historical_Cost_of_Ride', y='adjusted_ride_cost')
    plt.title('Adjusted Ride Cost vs. Historical Cost of Ride')
    plt.xlabel('Historical Cost of Ride')
    plt.ylabel('Adjusted Ride Cost')
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Checking for Outliers**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **IQR (Interquartile Range) method-univariate outliers**
    """)
    return


@app.cell
def _(df_org_2, numeric_cols_2, pd):
    outlier_results = []
    n_rows = len(df_org_2)
    for _col in numeric_cols_2:
        Q1 = df_org_2[_col].quantile(0.25)
        Q3 = df_org_2[_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df_org_2[(df_org_2[_col] < lower_bound) | (df_org_2[_col] > upper_bound)]
        n_outliers = len(outliers)
        pct_outliers = n_outliers / n_rows * 100
        outlier_results.append({'Variable': _col, 'Number_of_Outliers': n_outliers, 'Percentage_of_Outliers (%)': round(pct_outliers, 2)})
    outlier_summary_table = pd.DataFrame(outlier_results)
    outlier_summary_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Not many significant outliers have been observed in the dataset. So, we decided
    to keep the outliers as they are in the dataset.*
    *They are "Synthetic" Outliers*

    *Notice that Number_of_Riders and Number_of_Past_Rides have 0 outliers. The outliers only appear significantly in:*
    *adjusted_ride_cost (39 outliers)*
    *profit_percentage (41 outliers)*

    *This is because your formula created them. By multiplying the cost by demand and supply factors, you intentionally pushed some prices to extreme highs or lows. Deleting them would mean deleting the very logic you just built into the dataset.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Mahalanobis Distance-Mulivariate outliers**
    """)
    return


@app.cell
def _(df_org_2, inv, mahalanobis, np, numeric_cols_2):
    df_numeric = df_org_2[numeric_cols_2]
    cov_matrix = np.cov(df_numeric.values, rowvar=False)
    inv_cov_matrix = inv(cov_matrix)
    mean_numeric = np.mean(df_numeric.values, axis=0)
    mahalanobis_distances = []
    for _i in range(len(df_numeric)):
        distance = mahalanobis(df_numeric.iloc[_i].values, mean_numeric, inv_cov_matrix)
        mahalanobis_distances.append(distance)
    df_org_2['Mahalanobis_Distance'] = mahalanobis_distances
    return


@app.cell
def _(df_org_2, np):
    # Determine a threshold for outliers (based on chi-squared distribution)
    # For a significance level of 0.001 and 7 degrees of freedom (number of numeric columns), the chi-squared value is approximately 24.32
    # A common practice is to use a chi-squared distribution, but for simplicity, we can use a percentile for demonstration
    threshold = np.percentile(df_org_2['Mahalanobis_Distance'], 99)
    # Let's use the 99th percentile as a threshold for demonstration
    outliers_mahalanobis = df_org_2[df_org_2['Mahalanobis_Distance'] > threshold]
    print(f'Number of outliers detected using Mahalanobis Distance (threshold > {threshold:.2f}): {len(outliers_mahalanobis)}')
    # Identify outliers
    print(f'Percentage of outliers: {len(outliers_mahalanobis) / len(df_org_2) * 100:.2f}%')
    return outliers_mahalanobis, threshold


@app.cell
def _(display, outliers_mahalanobis):
    # Display the outliers
    display(outliers_mahalanobis.head())
    return


@app.cell
def _(df_org_2, plt, sns, threshold):
    plt.figure(figsize=(10, 6))
    sns.histplot(df_org_2['Mahalanobis_Distance'], kde=True)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Outlier Threshold ({threshold:.2f})')
    plt.title('Distribution of Mahalanobis Distances')
    plt.xlabel('Mahalanobis Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *The plot shows a right-skewed distribution typical of real-world economic data. Because the outliers follow the natural "decay" of the distribution and are not disconnected from the main body of data, they are statistically valid "extreme events" that your dynamic pricing model needs to understand.*
    """)
    return


@app.cell
def _(df_org_2):
    #dropping unwanted columns
    df_org_3 = df_org_2.drop(['Mahalanobis_Distance'], axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Does the adjusted ride cost differ across different location categories?**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check Normality by Group
    """)
    return


@app.cell
def _(df_train, shapiro):
    for loc in df_train['Location_Category'].unique():
        _stat, _p = shapiro(df_train[df_train['Location_Category'] == loc]['adjusted_ride_cost'])
        print(f'{loc} - W: {_stat:.3f}, p-value: {_p:.3g}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    adjusted_ride_cost is not normally distributed within any location category

    This violates the normality assumption of ANOVA
    """)
    return


@app.cell
def _(df_train, levene):
    # Group data by location category
    _groups = [df_train[df_train['Location_Category'] == loc]['adjusted_ride_cost'] for loc in df_train['Location_Category'].unique()]
    _stat, _p = levene(*_groups)
    print('Levene’s test statistic:', _stat)
    print('p-value:', _p)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    the variances of adjusted_ride_cost are not significantly different across locations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Homogeneity of variance assumption is satisfied, even though normality is violated.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Use non-parametric Kruskal-Wallis test**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    H₀: The distributions of adjusted ride cost are the same across location categories.

    H₁: At least one location category differs.
    """)
    return


@app.cell
def _(df_train, kruskal):
    _groups = [df_train[df_train['Location_Category'] == loc]['adjusted_ride_cost'] for loc in df_train['Location_Category'].unique()]
    _stat, _p = kruskal(*_groups)
    print('Kruskal-Wallis H-test statistic:', _stat)
    print('p-value:', _p)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fail to reject H₀ → no statistically significant difference in ride cost distributions between Rural, Suburban, and Urban locations
    """)
    return


if __name__ == "__main__":
    app.run()
