import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    #%pip install dfply kagglehub imblearn
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


app._unparsable_cell(
    r"""
    import kagglehub
    import os

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    # used for piping
    from dfply import *


    import warnings
    warnings.filterwarnings('ignore')
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('husl')
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Preperation
    """)
    return


@app.cell
def _(pd):
    # Load data
    # path = kagglehub.dataset_download("rasulmah/sri-lanka-weather-dataset")
    # df = pd.read_csv(os.path.join(path, "SriLanka_Weather_Dataset_V1.csv"))

    data_path = '../archive/SriLanka_Weather_Dataset_V1.csv'
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    print(f"Shape: {df.shape}, Cities: {df['city'].nunique()}")
    print(f"Date Range: {df['time'].min()} to {df['time'].max()}")
    return (df,)


@app.cell
def _(df, drop):
    df_1 = df >> drop('weathercode', 'sunrise', 'sunset', 'country', 'rain_sum', 'snowfall_sum')
    df_1.info()
    return (df_1,)


@app.cell
def _(df_1, train_test_split):
    train_df, test_df = train_test_split(df_1, test_size=0.2, random_state=42, stratify=df_1['city'])
    print(f'Training DataFrame shape: {train_df.shape}')
    print(f'Testing DataFrame shape: {test_df.shape}')
    return test_df, train_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Descriptive Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Patterns in Daily Precipitation
    """)
    return


@app.cell
def _(np, plt, train_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram of all precipitation
    ax1 = axes[0, 0]
    train_df['precipitation_sum'].hist(bins=50, ax=ax1, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_title('Distribution of Daily Precipitation', fontsize=12)
    ax1.set_xlabel('Precipitation (mm)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(train_df['precipitation_sum'].mean(), color='red', linestyle='--', label=f"Mean: {train_df['precipitation_sum'].mean():.2f}")
    ax1.legend()

    # Log-transformed histogram (non-zero values)
    ax2 = axes[0, 1]
    train_df_nonzero = train_df[train_df['precipitation_sum'] > 0]['precipitation_sum']
    ax2.hist(np.log1p(train_df_nonzero), bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_title('Distribution of Log(1 + Precipitation) - Non-zero Days', fontsize=12)
    ax2.set_xlabel('Log(1 + Precipitation)')
    ax2.set_ylabel('Frequency')

    # Box plot
    ax3 = axes[1, 0]
    train_df.boxplot(column='precipitation_sum', ax=ax3)
    ax3.set_title('Box Plot of Daily Precipitation', fontsize=12)
    ax3.set_ylabel('Precipitation (mm)')

    # KDE plot for non-zero precipitation
    ax4 = axes[1, 1]
    train_df_nonzero.plot.kde(ax=ax4, color='steelblue', linewidth=2)
    ax4.set_title('KDE of Non-Zero Precipitation', fontsize=12)
    ax4.set_xlabel('Precipitation (mm)')
    ax4.set_xlim(0, train_df_nonzero.quantile(0.99))

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(train_df):
    # Add temporal features
    train_df['year'] = train_df['time'].dt.year
    train_df['month'] = train_df['time'].dt.month
    train_df['day_of_year'] = train_df['time'].dt.dayofyear

    # Define Sri Lankan monsoon seasons
    def get_season(month):
        if month in [5, 6, 7, 8, 9]:  # Southwest Monsoon (Yala)
            return 'SW Monsoon (May-Sep)'
        elif month in [10, 11]:  # Second Inter-monsoon
            return 'Inter-monsoon 2 (Oct-Nov)'
        elif month in [12, 1, 2]:  # Northeast Monsoon (Maha)
            return 'NE Monsoon (Dec-Feb)'
        else:  # First Inter-monsoon
            return 'Inter-monsoon 1 (Mar-Apr)'

    train_df['season'] = train_df['month'].apply(get_season)
    print("Season distribution:")
    print(train_df['season'].value_counts())
    return


@app.cell
def _(plt, train_df):
    # Monthly and seasonal patterns
    fig_1, axes_1 = plt.subplots(1, 2, figsize=(14, 7))
    monthly_precip = train_df.groupby('month')['precipitation_sum'].mean()
    # Monthly average precipitation
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axes_1[0].bar(months, monthly_precip.values, color='steelblue', edgecolor='black')
    axes_1[0].set_title('Average Monthly Precipitation', fontsize=12)
    axes_1[0].set_ylabel('Precipitation (mm)')
    seasonal_precip = train_df.groupby('season')['precipitation_sum'].mean().sort_values(ascending=False)
    seasonal_precip.plot(kind='bar', ax=axes_1[1], color='coral', edgecolor='black')
    # Seasonal precipitation
    axes_1[1].set_title('Average Precipitation by Season', fontsize=12)
    axes_1[1].set_ylabel('Precipitation (mm)')
    axes_1[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(plt, train_df):
    # Precipitation intensity (mm per hour)
    train_df['precip_intensity'] = train_df.apply(lambda row: row['precipitation_sum'] / row['precipitation_hours'] if row['precipitation_hours'] > 0 else 0, axis=1)
    fig_2, axes_2 = plt.subplots(1, 2, figsize=(14, 5))
    train_df[train_df['precip_intensity'] > 0]['precip_intensity'].hist(bins=50, ax=axes_2[0], color='purple', alpha=0.7)
    axes_2[0].set_title('Distribution of Precipitation Intensity (Non-zero)', fontsize=12)
    axes_2[0].set_xlabel('Intensity (mm/hour)')
    axes_2[0].set_ylabel('Frequency')
    intensity_by_city = train_df[train_df['precip_intensity'] > 0].groupby('city')['precip_intensity'].mean().sort_values()
    # Distribution of intensity
    intensity_by_city.plot(kind='barh', ax=axes_2[1], color='purple', alpha=0.7)
    axes_2[1].set_title('Average Precipitation Intensity by City', fontsize=12)
    axes_2[1].set_xlabel('Intensity (mm/hour)')
    plt.tight_layout()
    # Intensity by city
    plt.show()
    return


@app.cell
def _(plt, sns, train_df):
    sns.boxplot(data=train_df['precipitation_sum'])
    plt.title('Box Plot of Variables')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploring Apparent Temperature
    """)
    return


@app.cell
def _(plt, train_df):
    fig_3, ax = plt.subplots(1, 1, figsize=(8, 6))
    train_df['temperature_2m_mean'].hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title('Distribution of Daily Mean Temperature', fontsize=12)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Frequency')
    ax.axvline(train_df['temperature_2m_mean'].mean(), color='red', linestyle='--', label=f'Mean: {train_df['temperature_2m_mean'].mean():.2f}')
    ax.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(plt, train_df):
    variables_to_plot = ['temperature_2m_mean', 'shortwave_radiation_sum', 'precipitation_sum', 'windspeed_10m_max']
    fig_4, axes_3 = plt.subplots(2, 2, figsize=(15, 10))
    axes_3 = axes_3.flatten()
    for i, var in enumerate(variables_to_plot):  # Flatten the 2x2 array of axes for easier iteration
        axes_3[i].hist(train_df[var], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes_3[i].set_title(f'Distribution of {var}', fontsize=12)
        axes_3[i].set_xlabel(var)
        axes_3[i].set_ylabel('Frequency')
        axes_3[i].axvline(train_df[var].mean(), color='red', linestyle='--', label=f'Mean: {train_df[var].mean():.2f}')
        axes_3[i].legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(plt, sns, train_df):
    sns.boxplot(data=train_df['temperature_2m_mean'])
    plt.title('Box Plot of Variables')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyzing Windspeed and Evapotranspiration
    """)
    return


@app.cell
def _(df_1):
    WEATHER = df_1.copy()
    print(WEATHER.head(5))
    return (WEATHER,)


@app.cell
def _(WEATHER, np, pd):
    WEATHER['time'] = pd.to_datetime(WEATHER['time'])
    WEATHER_1 = WEATHER.sort_values(by=['city', 'time']).reset_index(drop=True)
    cleaned_WEATHER = []
    for city_name, group in WEATHER_1.groupby('city'):
        group = group.copy()
        numeric_cols = group.select_dtypes(include=[np.number]).columns
        group[numeric_cols] = group[numeric_cols].interpolate(method='linear')
        group[numeric_cols] = group[numeric_cols].ffill().bfill()
        cleaned_WEATHER.append(group)
    WEATHER_1 = pd.concat(cleaned_WEATHER).reset_index(drop=True)
    return (WEATHER_1,)


@app.cell
def _(WEATHER_1):
    zone_map = {'Colombo': 'Wet Zone', 'Mount Lavinia': 'Wet Zone', 'Kesbewa': 'Wet Zone', 'Moratuwa': 'Wet Zone', 'Maharagama': 'Wet Zone', 'Ratnapura': 'Wet Zone', 'Galle': 'Wet Zone', 'Athurugiriya': 'Wet Zone', 'Weligama': 'Wet Zone', 'Matara': 'Wet Zone', 'Kolonnawa': 'Wet Zone', 'Gampaha': 'Wet Zone', 'Kalutara': 'Wet Zone', 'Bentota': 'Wet Zone', 'Mabole': 'Wet Zone', 'Hatton': 'Wet Zone', 'Oruwala': 'Wet Zone', 'Negombo': 'Wet Zone', 'Sri Jayewardenepura Kotte': 'Wet Zone', 'Kandy': 'Wet Zone', 'Jaffna': 'Dry Zone', 'Mannar': 'Dry Zone', 'Puttalam': 'Dry Zone', 'Trincomalee': 'Dry Zone', 'Kalmunai': 'Dry Zone', 'Hambantota': 'Dry Zone', 'Kurunegala': 'Intermediate Zone', 'Pothuhera': 'Intermediate Zone', 'Matale': 'Intermediate Zone', 'Badulla': 'Intermediate Zone'}
    WEATHER_1['zone'] = WEATHER_1['city'].map(zone_map)  # --- WET ZONE ---
    WEATHER_1.head()  # --- DRY ZONE ---  # --- INTERMEDIATE ZONE ---
    return (zone_map,)


@app.cell
def _(WEATHER_1):
    # Define the mapping of cities to their respective Wind Zones
    wind_zone_map = {'Jaffna': 'Zone I', 'Trincomalee': 'Zone I', 'Kalmunai': 'Zone I', 'Puttalam': 'Zone II', 'Mannar': 'Zone II', 'Kurunegala': 'Zone II', 'Pothuhera': 'Zone II', 'Matale': 'Zone II', 'Colombo': 'Zone III', 'Kandy': 'Zone III', 'Galle': 'Zone III', 'Mount Lavinia': 'Zone III', 'Kesbewa': 'Zone III', 'Moratuwa': 'Zone III', 'Maharagama': 'Zone III', 'Ratnapura': 'Zone III', 'Athurugiriya': 'Zone III', 'Weligama': 'Zone III', 'Matara': 'Zone III', 'Kolonnawa': 'Zone III', 'Gampaha': 'Zone III', 'Kalutara': 'Zone III', 'Bentota': 'Zone III', 'Mabole': 'Zone III', 'Hatton': 'Zone III', 'Oruwala': 'Zone III', 'Negombo': 'Zone III', 'Sri Jayewardenepura Kotte': 'Zone III', 'Hambantota': 'Zone III', 'Badulla': 'Zone III'}
    WEATHER_1['wind_zone'] = WEATHER_1['city'].map(wind_zone_map)  # Zone I: Northern & Eastern Coasts
    WEATHER_1['wind_zone'] = WEATHER_1['wind_zone'].fillna('Unknown')
    print(WEATHER_1.head())  # Eastern Coast  # Zone II: Intermediate areas  # Intermediate climatic zone  # Zone III: Southern & Western areas  # Southern Coast  # Hilly/Southern Central, aligns with Zone III
    return (wind_zone_map,)


@app.cell
def _(WEATHER_1, wind_zone_map, zone_map):
    import numpy as np
    import matplotlib.pyplot as plt
    WEATHER_with_wind_dir = WEATHER_1.copy()
    WEATHER_with_wind_dir['zone'] = WEATHER_with_wind_dir['city'].map(zone_map)
    WEATHER_with_wind_dir['wind_zone'] = WEATHER_with_wind_dir['city'].map(wind_zone_map).fillna('Unknown')

    def plot_wind_rose(data, title, color='skyblue'):
        num_bins = 16
        bins = np.linspace(0, 360, num_bins + 1)
        bin_centers = np.deg2rad(bins[:-1] + np.diff(bins) / 2)
        counts, _ = np.histogram(data['winddirection_10m_dominant'], bins=bins)
        frequency = counts / counts.sum() if counts.sum() > 0 else counts
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
        ax.bar(bin_centers, frequency, width=np.deg2rad(360 / num_bins), bottom=0.0, color=color, edgecolor='black', alpha=0.7)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        max_freq = max(frequency) if len(frequency) > 0 else 0.1
        ticks = np.linspace(0, max_freq, 5)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{y * 100:.1f}%' for y in ticks])
        plt.title(title, pad=20, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    plot_wind_rose(WEATHER_with_wind_dir, 'Overall Dominant Wind Direction Frequency (Sri Lanka)')  # Clockwise
    colors = {'Zone 1': '#3498db', 'Zone 2': '#e67e22', 'Zone 3': '#2ecc71', 'Unknown': '#95a5a6'}
    for w_zone in WEATHER_with_wind_dir['wind_zone'].unique():
        subset = WEATHER_with_wind_dir[WEATHER_with_wind_dir['wind_zone'] == w_zone]
        if not subset.empty:
            plot_wind_rose(subset, f'Wind Direction Frequency: {w_zone}', color=colors.get(w_zone, 'skyblue'))
    return WEATHER_with_wind_dir, np, plt


@app.cell
def _(WEATHER_with_wind_dir, pd, plt):
    import seaborn as sns

    def plot_monthly_wind_speed(df):
        print(f"\n" + "="*60)
        print(f"ANALYZING MONTHLY WIND SPEED TRENDS BY ZONE")
        print("="*60)


        df['time'] = pd.to_datetime(df['time'])
        df['month'] = df['time'].dt.month


        monthly_stats = df.groupby(['wind_zone', 'month'])['windspeed_10m_max'].mean().reset_index()

        plt.figure(figsize=(12, 6))


        sns.lineplot(data=monthly_stats, x='month', y='windspeed_10m_max',
                     hue='wind_zone', marker='o', linewidth=2.5)


        plt.title('Average Monthly Wind Speed by Wind Zone', fontsize=15, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Average Wind Speed (m/s)', fontsize=12)
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Wind Zone', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()


        pivot_table = monthly_stats.pivot(index='month', columns='wind_zone', values='windspeed_10m_max')
        print("\nSummary Table (Average Speed per Month):")
        print(pivot_table)


    plot_monthly_wind_speed(WEATHER_with_wind_dir)
    return (sns,)


@app.cell
def _(WEATHER_1, plt, sns, zone_map):
    import pandas as pd

    def map_evaporation_by_city(df):
        print(f'\n' + '=' * 60)
        print(f'MAPPING AVERAGE EVAPOTRANSPIRATION BY CITY')
        print('=' * 60)
        city_stats = df.groupby(['zone', 'city'])['et0_fao_evapotranspiration'].mean().reset_index()
        city_stats = city_stats.sort_values(by=['zone', 'et0_fao_evapotranspiration'], ascending=[True, False])
        plt.figure(figsize=(12, 10))
        sns.barplot(data=city_stats, x='et0_fao_evapotranspiration', y='city', hue='zone', dodge=False, palette={'Wet Zone': '#2ecc71', 'Intermediate Zone': '#f1c40f', 'Dry Zone': '#e74c3c'})
        plt.title('Average Daily Evapotranspiration (ET0) by City & Zone', fontsize=16, fontweight='bold')
        plt.xlabel('Average ET0 (mm/day)', fontsize=12)
        plt.ylabel('City', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.legend(title='Climatic Zone', loc='lower right')
        for index, value in enumerate(city_stats['et0_fao_evapotranspiration']):
            plt.text(value + 0.05, index, f'{value:.2f}', va='center', fontsize=9)
        plt.tight_layout()
        plt.show()
        return city_stats
    if 'zone' not in WEATHER_1.columns:
        WEATHER_1['zone'] = WEATHER_1['city'].map(zone_map)
    avg_evap_stats = map_evaporation_by_city(WEATHER_1)
    return (pd,)


@app.cell
def _(WEATHER_1, WEATHER_with_wind_dir, np, plt, zone_map):
    import matplotlib.patches as patches
    city_coords = {'Jaffna': [9.6615, 80.0255], 'Mannar': [8.9766, 79.9043], 'Vavuniya': [8.7514, 80.4971], 'Trincomalee': [8.5874, 81.2152], 'Anuradhapura': [8.3114, 80.4037], 'Puttalam': [8.0408, 79.8394], 'Polonnaruwa': [7.9403, 81.0188], 'Batticaloa': [7.731, 81.6747], 'Kurunegala': [7.4863, 80.3647], 'Matale': [7.4726, 80.6234], 'Kandy': [7.2906, 80.6337], 'Negombo': [7.2008, 79.8737], 'Gampaha': [7.084, 79.9939], 'Colombo': [6.9271, 79.8612], 'Sri Jayewardenepura Kotte': [6.8863, 79.9187], 'Nuwara Eliya': [6.9497, 80.7891], 'Badulla': [6.9934, 81.055], 'Ratnapura': [6.6828, 80.3992], 'Kalutara': [6.5854, 79.9607], 'Bentota': [6.4223, 80.0056], 'Galle': [6.0535, 80.221], 'Matara': [5.9549, 80.555], 'Hambantota': [6.1429, 81.1212], 'Ampara': [7.2825, 81.6667], 'Monaragala': [6.8714, 81.3487]}
    sl_outline = [(79.7, 9.8), (80.1, 9.8), (80.3, 9.5), (80.5, 9.2), (80.8, 9.3), (81.0, 9.0), (81.2, 8.6), (81.3, 8.2), (81.5, 7.8), (81.7, 7.5), (81.8, 7.0), (81.8, 6.5), (81.5, 6.2), (81.2, 6.0), (80.8, 5.9), (80.5, 5.8), (80.2, 5.9), (80.0, 6.2), (79.9, 6.5), (79.8, 6.8), (79.8, 7.2), (79.7, 7.8), (79.7, 8.2), (79.8, 8.6), (79.7, 9.0), (79.7, 9.8)]

    def plot_custom_map(df):
        if 'zone' not in df.columns:
            df['zone'] = df['city'].map(zone_map)
        map_data = df.groupby(['city', 'zone'])['et0_fao_evapotranspiration'].mean().reset_index()
        map_data['lat'] = map_data['city'].map(lambda x: city_coords.get(x, [np.nan, np.nan])[0])
        map_data['lon'] = map_data['city'].map(lambda x: city_coords.get(x, [np.nan, np.nan])[1])
        map_data = map_data.dropna(subset=['lat', 'lon'])
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.set_facecolor('#e0f7fa')
        poly = patches.Polygon(sl_outline, closed=True, facecolor='#fdfefe', edgecolor='#7f8c8d', linewidth=1.5)
        ax.add_patch(poly)
        colors = {'Wet Zone': '#2ecc71', 'Intermediate Zone': '#f39c12', 'Dry Zone': '#e74c3c'}
        for zone, color in colors.items():
            subset = map_data[map_data['zone'] == zone]
            if subset.empty:
                continue
            ax.scatter(subset['lon'], subset['lat'], s=subset['et0_fao_evapotranspiration'] ** 3.5 * 2, color=color, alpha=0.8, edgecolor='black', linewidth=1, label=zone, zorder=5)
        for _, row in map_data.iterrows():
            x_offset = 0.08
            ha = 'left'
            if row['lon'] > 81.0:  # North (Jaffna/Mullaitivu)
                x_offset = -0.08  # East Coast
                ha = 'right'  # South East
            ax.text(row['lon'] + x_offset, row['lat'], f'{row['city']}\n{row['et0_fao_evapotranspiration']:.1f}', fontsize=9, va='center', ha=ha, fontweight='bold', color='#2c3e50', zorder=10)  # South West
        ax.set_title('Average Evapotranspiration (ET0) Map', fontsize=16, fontweight='bold')  # West Coast
        ax.set_xlim(79.0, 82.2)
        ax.set_ylim(5.5, 10.0)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', color='white', alpha=0.5)
        legend = ax.legend(title='Climatic Zone', loc='lower right')
        legend.get_frame().set_alpha(0.9)
        plt.tight_layout()
        plt.show()
    if 'WEATHER_with_wind_dir' in locals():
        plot_custom_map(WEATHER_with_wind_dir)
    else:
        plot_custom_map(WEATHER_1)  # Ocean Color  # Bubble size
    return


@app.cell
def _(WEATHER_1, df_weather, plt):
    weather_cols = ['et0_fao_evapotranspiration', 'precipitation_sum']
    if 'WEATHER' in locals():
        zonal_daily = WEATHER_1.groupby(['time', 'zone'])[weather_cols].mean(numeric_only=True).unstack()
        zonal_daily.columns = [f'{col[0]}_{col[1]}' for col in zonal_daily.columns]
    elif 'df_weather' in locals():
    # Fix: Define zonal_daily derived from WEATHER dataframe
    # This variable was missing in the original cell causing a NameError
        zonal_daily = df_weather.groupby(['time', 'zone'])[weather_cols].mean(numeric_only=True).unstack()
        zonal_daily.columns = [f'{col[0]}_{col[1]}' for col in zonal_daily.columns]
    else:
        print('Error: WEATHER dataframe not found for zonal_daily calculation')
    annual_rain = zonal_daily.resample('YE').sum()
    baseline_rain = annual_rain['precipitation_sum_Wet Zone'].mean()
    plt.figure(figsize=(14, 7))
    target_years = [2016, 2017, 2020, 2023]
    colors_1 = ['#e74c3c' if year in target_years else '#3498db' for year in annual_rain.index.year]
    bars = plt.bar(annual_rain.index.year, annual_rain['precipitation_sum_Wet Zone'], color=colors_1, alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.axhline(baseline_rain, color='black', linestyle='--', label=f'Baseline Avg: {baseline_rain:.0f}mm')
    plt.title('Wet Zone Annual Rainfall: Deficit Analysis (2010-2025)', fontsize=15)
    plt.ylabel('Total Annual Rainfall (mm)', fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.xticks(annual_rain.index.year, rotation=45)
    plt.legend(loc='upper left')
    plt.grid(axis='y', alpha=0.2)
    for bar, year in zip(bars, annual_rain.index.year):
        if year in target_years:
            val = bar.get_height()
            perc = (val / baseline_rain - 1) * 100
            plt.text(bar.get_x() + bar.get_width() / 2, val + 20, f'{perc:+.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#c0392b')
    plt.tight_layout()
    plt.show()
    print(f'{'Year':<6} | {'Rainfall (mm)':<15} | {'Deficit %':<18}')
    print('-' * 45)
    for year in sorted(target_years):
        if year in annual_rain.index.year:
            actual = annual_rain.loc[f'{year}-12-31', 'precipitation_sum_Wet Zone']
            diff_perc = (actual / baseline_rain - 1) * 100
            print(f'{year:<6} | {actual:<15.2f} | {diff_perc:+.2f}%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Advance Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objective 1: Understanding and Predicting Daily Precipitation
    """)
    return


@app.cell
def _(plt, sns, train_df):
    sns.scatterplot(data=train_df, x='precipitation_hours', y='precipitation_sum')
    plt.title('Precipitation Hours vs Total Precipitation')
    plt.show()
    return


@app.cell
def _(plt, train_df):
    import matplotlib.dates as mdates
    df_time_indexed = train_df.set_index('time')

    annual_precip = df_time_indexed.resample('YE')['precipitation_sum'].sum()

    plt.figure(figsize=(12, 6))
    plt.plot(annual_precip.index, annual_precip.values, marker='o', linestyle='-', markersize=4, alpha=0.7)

    plt.title('Annual Precipitation Totals Over Time (Yearly Spacing)')
    plt.xlabel('Year')
    plt.ylabel('Total Precipitation (mm)')
    plt.grid(True)


    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.show()
    return df_time_indexed, mdates


@app.cell
def _(df_time_indexed, plt):
    monthly_totals = df_time_indexed['precipitation_sum'].resample('MS').sum()

    plt.figure(figsize=(15, 7))
    plt.plot(monthly_totals.index, monthly_totals.values, marker='o', linestyle='-')
    plt.title('Monthly Precipitation Totals Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Precipitation (mm)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mdates, plt, train_df):
    from statsmodels.tsa.seasonal import seasonal_decompose
    wet_cities = ['Colombo', 'Galle', 'Ratnapura', 'Kalutara']
    dry_cities = ['Jaffna', 'Trincomalee', 'Anuradhapura', 'Vavuniya']

    def get_decomposition(city_list, label):
        zone_data = train_df[train_df['city'].isin(city_list)].copy()
        zone_avg = zone_data.groupby('time')['precipitation_sum'].mean()  # Filter for cities in the list
        monthly_avg = zone_avg.resample('ME').mean()
        return seasonal_decompose(monthly_avg, model='additive', period=12)
    res_wet = get_decomposition(wet_cities, 'Wet Zone')  # Group by Time (Average all cities in the zone together)
    res_dry = get_decomposition(dry_cities, 'Dry Zone')
    fig_5, axes_4 = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
    axes_4[0, 0].set_title('Wet Zone (South-West Monsoon)', fontsize=14, fontweight='bold', color='navy')  # Resample to Monthly
    axes_4[0, 1].set_title('Dry Zone (North-East Monsoon)', fontsize=14, fontweight='bold', color='darkred')
    res_wet.observed.plot(ax=axes_4[0, 0], color='navy', legend=False)
    axes_4[0, 0].set_ylabel('Observed')  # Decompose
    res_wet.trend.plot(ax=axes_4[1, 0], color='navy', legend=False)
    axes_4[1, 0].set_ylabel('Trend')
    res_wet.seasonal.plot(ax=axes_4[2, 0], color='navy', legend=False)
    axes_4[2, 0].set_ylabel('Seasonal')
    res_wet.resid.plot(ax=axes_4[3, 0], color='navy', legend=False)
    axes_4[3, 0].set_ylabel('Residual')
    res_dry.observed.plot(ax=axes_4[0, 1], color='darkred', legend=False)
    res_dry.trend.plot(ax=axes_4[1, 1], color='darkred', legend=False)
    res_dry.seasonal.plot(ax=axes_4[2, 1], color='darkred', legend=False)
    res_dry.resid.plot(ax=axes_4[3, 1], color='darkred', legend=False)
    plt.tight_layout()
    for ax_1 in axes_4[3, :]:
        ax_1.xaxis.set_major_locator(mdates.YearLocator())
    # --- Plot Wet Zone (Left Column) ---
        ax_1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # --- Plot Dry Zone (Right Column) ---
    plt.show()
    return (seasonal_decompose,)


@app.cell
def _(df_1, test_df, train_df):
    df_1['rain_occurrence'] = (df_1['precipitation_sum'] > 0).astype(int)
    train_df['rain_occurrence'] = (train_df['precipitation_sum'] > 0).astype(int)
    test_df['rain_occurrence'] = (test_df['precipitation_sum'] > 0).astype(int)
    return


@app.cell
def _(display, np, plt, sns, train_df):
    numerical_cols = train_df.select_dtypes(include=np.number).columns.tolist()

    if 'log_precipitation_sum' in numerical_cols:
        numerical_cols.remove('log_precipitation_sum')

    correlation_matrix = train_df[numerical_cols].corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Weather Features')
    plt.show()

    print("\nCorrelations with Precipitation Sum:")
    display(correlation_matrix['precipitation_sum'].sort_values(ascending=False))

    print("\nCorrelations with Rain Occurrence:")
    display(correlation_matrix['rain_occurrence'].sort_values(ascending=False))
    return


@app.cell
def _(plt, sns, train_df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=train_df['shortwave_radiation_sum'], y=train_df['precipitation_sum'], alpha=0.6)
    plt.title('Precipitation Sum vs. Shortwave Radiation Sum')
    plt.xlabel('Shortwave Radiation Sum')
    plt.ylabel('Precipitation Sum (mm)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(plt, sns, train_df):
    rain_counts = train_df['rain_occurrence'].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    sns.barplot(x=rain_counts.index, y=rain_counts.values, palette='viridis')
    plt.title('Count of Rain Occurrence')
    plt.xlabel('Rain Occurrence (0: No Rain, 1: Rain)')
    plt.ylabel('Number of Days')
    plt.xticks([0, 1], ['No Rain', 'Rain'])
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(plt, sns, train_df):
    import scipy.stats as stats
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    def get_season_1(month):
        if month in [3, 4]:
            return '1st Inter-monsoon'
        elif month in [5, 6, 7, 8, 9]:
            return 'SW Monsoon'
        elif month in [10, 11]:
            return '2nd Inter-monsoon'
        else:
            return 'NE Monsoon'  # Dec, Jan, Feb
    train_df['season'] = train_df['time'].dt.month.apply(get_season_1)
    groups = [train_df[train_df['season'] == s]['precipitation_sum'] for s in train_df['season'].unique()]
    f_stat, p_value = stats.f_oneway(*groups)
    print('=== One-Way ANOVA Results ===')
    print(f'F-Statistic: {f_stat:.4f}')
    print(f'P-Value: {p_value:.4e}')
    if p_value < 0.05:
        print("\nSignificant difference found. Running Tukey's HSD...")
        tukey = pairwise_tukeyhsd(endog=train_df['precipitation_sum'], groups=train_df['season'], alpha=0.05)
        print(tukey)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='season', y='precipitation_sum', data=train_df, showfliers=False)
    plt.title('Daily Precipitation Distribution by Season')
    plt.ylabel('Precipitation (mm)')
    plt.show()
    return pairwise_tukeyhsd, stats


@app.cell
def _(df_1):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(df_1['precipitation_sum'].dropna())
    adf_statistic = result[0]
    p_value_1 = result[1]
    critical_values = result[4]
    print('=== Augmented Dickey-Fuller Test Results ===')
    print(f'ADF Statistic: {adf_statistic:.4f}')
    print(f'P-Value: {p_value_1:.4e}')
    print('Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key}: {value:.4f}')
    if p_value_1 < 0.05:
        print('\nResult: STATIONARY (Reject H0)')
        print('The rainfall data does not have a trend. The mean and variance are constant over time.')
    else:
        print('\nResult: NON-STATIONARY (Fail to Reject H0)')
        print("The data has a trend or seasonality. You might need to 'difference' the data.")
    return (adfuller,)


@app.cell
def _(df_1, pd):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    df_2 = df_1.sort_values(by=['city', 'time'])
    #Create Lags
    df_2['rain_lag_1'] = df_2.groupby('city')['precipitation_sum'].shift(1)
    df_2['rain_lag_2'] = df_2.groupby('city')['precipitation_sum'].shift(2)
    df_2['rain_lag_3'] = df_2.groupby('city')['precipitation_sum'].shift(3)
    features = ['rain_lag_1', 'rain_lag_2', 'rain_lag_3', 'apparent_temperature_mean', 'windspeed_10m_max', 'winddirection_10m_dominant', 'shortwave_radiation_sum', 'latitude', 'longitude', 'elevation']
    df_model = df_2.dropna(subset=features + ['rain_occurrence']).copy()
    X = df_model[features]
    y = df_model['rain_occurrence']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    print('Final Optimized Feature Importance (with Apparent Temp Mean):')
    importance = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_[0]}).sort_values(by='Coefficient', ascending=False)
    print(importance)
    print('\nModel Performance Metrics:')
    print(classification_report(y_test, model.predict(X_test_scaled)))
    return LogisticRegression, df_2, features, model, train_test_split


@app.cell
def _(features, model, np, pd):
    importance_1 = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_[0]})
    importance_1['Odds_Ratio'] = np.exp(importance_1['Coefficient'])
    importance_1['Abs_Coefficient'] = importance_1['Coefficient'].abs()
    importance_1 = importance_1.sort_values(by='Abs_Coefficient', ascending=False)
    print(importance_1.drop(columns=['Abs_Coefficient']))
    return (importance_1,)


@app.cell
def _(importance_1, plt, sns):
    plt.figure(figsize=(10, 6))
    colors_2 = ['#1f77b4' if x > 0 else '#d62728' for x in importance_1['Coefficient']]
    sns.barplot(x='Coefficient', y='Feature', data=importance_1, hue='Feature', palette=colors_2, legend=False)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.title('Feature Importance: What drives rainfall in Sri Lanka?')
    plt.xlabel('Coefficient Strength (Log-Odds)')
    plt.ylabel('Weather Variable')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objective 2: Modeling Apparent Temperature and Thermal Comfort
    """)
    return


@app.cell
def _(plt, sns):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    import warnings
    warnings.filterwarnings('ignore')
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('husl')
    return (
        LinearRegression,
        Pipeline,
        RandomForestRegressor,
        StandardScaler,
        mean_squared_error,
        r2_score,
        warnings,
    )


@app.cell
def _(df_2):
    df_2.duplicated().any()
    return


@app.cell
def _(df_2, train_test_split):
    # train and test data split
    train_df_apt, test_df_apt = train_test_split(df_2, test_size=0.2, random_state=42, stratify=df_2['city'])
    print(f'Training DataFrame shape: {train_df_apt.shape}')
    print(f'Testing DataFrame shape: {test_df_apt.shape}')
    return test_df_apt, train_df_apt


@app.cell
def _(plt, sns, train_df_apt):
    # Correlation heatmap of the apparent temparature and its associated variables.
    cols_for_corr = [
        'apparent_temperature_mean', 'temperature_2m_mean',
        'et0_fao_evapotranspiration', 'shortwave_radiation_sum',
        'windspeed_10m_max', 'elevation'
    ]

    corr_matrix = train_df_apt[cols_for_corr].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap: Thermal Comfort Factors')
    plt.show()
    return


@app.cell
def _(train_df_apt):
    # grouping mean temparature and mean temparature of each city
    city_stats=train_df_apt.groupby('city')[['temperature_2m_mean','apparent_temperature_mean']].mean().reset_index()
    city_stats['temp_diff'] = city_stats['apparent_temperature_mean'] - city_stats['temperature_2m_mean']
    city_stats=city_stats.sort_values(by='temp_diff',ascending=False)
    city_stats
    return (city_stats,)


@app.cell
def _(city_stats, plt):
    # plotting mean of actual and aparent temp. of each city
    plt.figure(figsize=(14, 7))
    x = range(len(city_stats))
    width = 0.35

    plt.bar([i - width/2 for i in x], city_stats['temperature_2m_mean'], width, label='Actual Temp', color='skyblue')
    plt.bar([i + width/2 for i in x], city_stats['apparent_temperature_mean'], width, label='Apparent (Feel-Like)', color='coral')

    plt.xticks(x, city_stats['city'], rotation=90)
    plt.ylabel('Temperature (°C)')
    plt.title('Actual vs. Apparent Mean Temperature across Sri Lankan Cities')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    return


@app.cell
def _(city_stats, plt, sns):
    #plotting temp. difference of each city

    plt.figure(figsize=(14, 6))
    sns.barplot(x='city', y='temp_diff', data=city_stats, palette='YlOrRd_r')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title('The "Heat Index" Effect (Apparent - Actual Temperature)')
    plt.ylabel('Difference (°C)')
    plt.xticks(rotation=90)
    plt.show()
    return


@app.cell
def _(plt, train_df_apt):
    ## annual cycle of actual and aparent temp.
    train_df_apt['month'] = train_df_apt['time'].dt.month
    monthly_cycle = train_df_apt.groupby('month')[['temperature_2m_mean', 'apparent_temperature_mean']].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(monthly_cycle.index, monthly_cycle['temperature_2m_mean'], marker='o', label='Actual Temp')
    plt.plot(monthly_cycle.index, monthly_cycle['apparent_temperature_mean'], marker='s', label='Apparent Temp')
    plt.title('Annual Cycle of Actual vs. Apparent Temperature')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.show()
    return


@app.cell
def _(train_df):
    #elevation of each city
    city_elevation_list = train_df[['city', 'elevation']].drop_duplicates().sort_values(by='elevation', ascending=True)
    city_elevation_list = city_elevation_list.reset_index(drop=True)
    print(city_elevation_list)
    return


@app.cell
def _(train_df_apt):
    # grouping cities accriding to their elevation(low,mid,high)
    train_df_apt['temp_diff'] = train_df_apt['apparent_temperature_mean'] - train_df_apt['temperature_2m_mean']

    def get_elevation_level(elev):
        if elev < 300:
            return 'Low'
        elif elev < 900:
            return 'Mid'
        else:
            return 'High'

    train_df_apt['elevation_level'] = train_df_apt['elevation'].apply(get_elevation_level)

    print("Data distribution by Elevation Level:")
    print(train_df_apt['elevation_level'].value_counts())

    train_df_apt[['city', 'elevation', 'elevation_level', 'temp_diff']].head()
    return


@app.cell
def _(plt, sns, train_df_apt):
    # temp. difference distributiin of each elevation level
    plt.figure(figsize=(10, 6))

    sns.boxplot(x='elevation_level', y='temp_diff', data=train_df_apt,
                order=['Low', 'Mid', 'High'], palette='coolwarm')

    plt.title('Distribution of Thermal Comfort Gap across Elevation Levels')
    plt.xlabel('Elevation Level (Low < 300m | Mid 300-900m | High > 900m)')
    plt.ylabel('Apparent - Actual Temperature (°C)')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()
    return


@app.cell
def _(plt, sns, train_df_apt):
    # distribution of aparent temp.
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df_apt['apparent_temperature_mean'], kde=True, color='teal', bins=30)
    plt.title('Frequency Distribution of Apparent Temperature in Sri Lanka', fontsize=14)
    plt.xlabel('Apparent Temperature (°C)')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=train_df_apt, x='apparent_temperature_mean', hue='elevation_level',
                hue_order=['Low', 'Mid', 'High'], fill=True, palette='coolwarm', alpha=0.5)
    plt.title('Apparent Temperature Distribution Across Elevation Zones', fontsize=14)
    plt.xlabel('Apparent Temperature (°C)')
    plt.ylabel('Density')
    plt.show()
    return


@app.cell
def _(stats, train_df_apt):
    ## performing one way anova
    low_group = train_df_apt[train_df_apt['elevation_level'] == 'Low']['temp_diff']
    mid_group = train_df_apt[train_df_apt['elevation_level'] == 'Mid']['temp_diff']
    high_group = train_df_apt[train_df_apt['elevation_level'] == 'High']['temp_diff']
    f_stat_1, p_val = stats.f_oneway(low_group, mid_group, high_group)
    print(f'--- ANOVA Results ---')
    print(f'F-Statistic: {f_stat_1:.2f}')
    print(f'P-Value: {p_val:.4e}')
    alpha = 0.05
    if p_val < alpha:
        print('\nResult: Reject the Null Hypothesis.')
        print('Conclusion: Elevation significantly impacts the difference between Actual and Apparent temperature.')
    else:
        print('\nResult: Fail to reject the Null Hypothesis.')
        print('Conclusion: No statistically significant difference was found.')
    return


@app.cell
def _(pairwise_tukeyhsd, train_df_apt):
    # Perform Tukey HSD test
    tukey_1 = pairwise_tukeyhsd(endog=train_df_apt['temp_diff'], groups=train_df_apt['elevation_level'], alpha=0.05)
    print(tukey_1)
    return (tukey_1,)


@app.cell
def _(plt, tukey_1):
    # Plotting tukey results
    tukey_1.plot_simultaneous()
    plt.vlines(x=0, ymin=-0.5, ymax=2.5, color='red', linestyle='--')
    plt.title('Tukey HSD Comparison of Elevation Levels')
    plt.xlabel('Mean Difference in Temperature Gap (°C)')
    plt.show()
    return


@app.cell
def _(pd, train_df_apt):
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    features_1 = ['temperature_2m_mean', 'windspeed_10m_max', 'shortwave_radiation_sum', 'et0_fao_evapotranspiration', 'elevation']
    X_1 = train_df_apt[features_1]
    X_with_const = sm.add_constant(X_1)
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X_with_const.columns
    vif_data['VIF'] = [variance_inflation_factor(X_with_const.values, i) for i in range(len(X_with_const.columns))]
    print(vif_data[vif_data['Feature'] != 'const'].sort_values(by='VIF', ascending=False))
    return (X_1,)


@app.cell
def _():
    features_with_et0 = ['temperature_2m_mean', 'windspeed_10m_max', 'shortwave_radiation_sum', 'elevation', 'et0_fao_evapotranspiration']
    features_without_et0 = ['temperature_2m_mean', 'windspeed_10m_max', 'shortwave_radiation_sum', 'elevation']

    target = 'apparent_temperature_mean'
    return features_with_et0, features_without_et0, target


@app.cell
def _(
    LinearRegression,
    Pipeline,
    StandardScaler,
    features_with_et0,
    features_without_et0,
    metrics,
    np,
    pd,
    target,
    test_df_apt,
    train_df_apt,
):
    # Helper function to train and evaluate
    def train_and_evaluate(feature_list, name):
        # Prepare data
        X_train_apt = train_df_apt[feature_list]
        y_train_apt = train_df_apt[target]
        X_test_apt = test_df_apt[feature_list]
        y_test_apt = test_df_apt[target]

        # Using Pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        pipe.fit(X_train_apt, y_train_apt)

        # Prediction and evaluate
        preds = pipe.predict(X_test_apt)
        return {
            "Model": name,
            "R2": metrics.r2_score(y_test_apt, preds),
            "RMSE": np.sqrt(metrics.mean_squared_error(y_test_apt, preds)),
            "Pipeline": pipe,
            "Predictions": preds
        }


    # create 2 models.
    results_et0 = train_and_evaluate(features_with_et0, "With ET0")
    results_no_et0 = train_and_evaluate(features_without_et0, "Without ET0")

    # this table compares two models
    comparison_df = pd.DataFrame([results_et0, results_no_et0]).drop(columns=['Pipeline', 'Predictions'])
    print(comparison_df)
    return (results_et0,)


@app.cell
def _(features_with_et0, plt, results_et0, stats, target, test_df_apt):
    #Q-Q plot
    X_test_apt = test_df_apt[features_with_et0]
    y_test_apt = test_df_apt[target]
    y_pred_apt = results_et0['Predictions']

    residuals = y_test_apt - y_pred_apt
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot: Assessing Residual Normality', fontsize=14)
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Ordered Values (Residuals)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    return residuals, y_pred_apt, y_test_apt


@app.cell
def _(plt, residuals, sns, y_pred_apt):
    # Residual vs fitted values plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred_apt, y=residuals, alpha=0.3, color='orange')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs. Fitted Values (Homoscedasticity Check)')
    plt.xlabel('Predicted Apparent Temperature')
    plt.ylabel('Residuals')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Forest Approach
    """)
    return


@app.cell
def _(
    RandomForestRegressor,
    features_with_et0,
    metrics,
    np,
    target,
    test_df_apt,
    train_df_apt,
    y_test_apt,
):
    rf_model_apt = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_apt.fit(train_df_apt[features_with_et0], train_df_apt[target])

    rf_preds_apt = rf_model_apt.predict(test_df_apt[features_with_et0])
    print(f"\nRandom Forest R2: {metrics.r2_score(y_test_apt, rf_preds_apt):.4f}")
    print(f"Random Forest RMSE: {np.sqrt(metrics.mean_squared_error(y_test_apt, rf_preds_apt)):.4f} °C")
    return (rf_model_apt,)


@app.cell
def _(features_with_et0, pd, plt, rf_model_apt, sns):
    ## plotting feature importance graph.
    importances = rf_model_apt.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': features_with_et0,
        'Importance': importances
    })
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
    plt.title('Feature Importance: Random Forest Model', fontsize=14)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objective 3:Analyzing Wind Dynamics and Evapotranspiration
    """)
    return


@app.cell
def _(WEATHER_1, plt, sns):
    # Group by time and zone, then calculate the mean et0_fao_evapotranspiration
    eva_by_zone = WEATHER_1.groupby(['time', 'zone'])['et0_fao_evapotranspiration'].mean().reset_index()
    eva_by_zone_smoothed = eva_by_zone.groupby('zone')['et0_fao_evapotranspiration'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    # Apply a rolling mean for smoothing within each zone
    eva_by_zone['et0_fao_evapotranspiration_smoothed'] = eva_by_zone_smoothed
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=eva_by_zone, x='time', y='et0_fao_evapotranspiration_smoothed', hue='zone')
    plt.title('Smoothed Time Series of Mean et0_fao_evapotranspiration by Climatic Zone (30-day rolling mean)')
    plt.xlabel('Date')
    plt.ylabel('Smoothed Mean et0_fao_evapotranspiration')
    plt.grid(True)
    plt.legend(title='Zone')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(WEATHER_1, plt, sns):
    # Group by time and wind_zone, then calculate the mean windspeed_10m_max
    wind_speed_by_zone_agg = WEATHER_1.groupby(['time', 'wind_zone'])['windspeed_10m_max'].mean().reset_index()
    wind_speed_by_zone_agg['windspeed_10m_max_smoothed'] = wind_speed_by_zone_agg.groupby('wind_zone')['windspeed_10m_max'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    # Apply a rolling mean for smoothing within each wind zone
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=wind_speed_by_zone_agg, x='time', y='windspeed_10m_max_smoothed', hue='wind_zone')
    plt.title('Smoothed Time Series of Mean Maximum Wind Speed by Wind Zone (30-day rolling mean)')
    plt.xlabel('Date')
    plt.ylabel('Smoothed Mean Maximum Wind Speed (m/s)')
    plt.grid(True)
    plt.legend(title='Wind Zone')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(WEATHER_1, pd, plt, seasonal_decompose):
    evap_monthly = WEATHER_1.groupby([pd.Grouper(key='time', freq='ME'), 'zone'])['et0_fao_evapotranspiration'].mean().unstack()
    wind_monthly = WEATHER_1.groupby([pd.Grouper(key='time', freq='ME'), 'wind_zone'])['windspeed_10m_max'].mean().unstack()

    def plot_decomposition(data, title):
        for zone in data.columns:
            print(f'Decomposing {title} for {zone}...')
            res = seasonal_decompose(data[zone].dropna(), model='additive', period=12)
            fig = res.plot()
            fig.set_size_inches(12, 8)
            plt.suptitle(f'{title} Decomposition: {zone}', fontsize=16)
            plt.show()
    plot_decomposition(evap_monthly, 'Evaporation')
    plot_decomposition(wind_monthly, 'Wind Speed')
    return


@app.cell
def _(WEATHER_1, pd, plt, seasonal_decompose):
    evap_monthly_1 = WEATHER_1.groupby([pd.Grouper(key='time', freq='ME'), 'zone'])['et0_fao_evapotranspiration'].mean().unstack()
    wind_monthly_1 = WEATHER_1.groupby([pd.Grouper(key='time', freq='ME'), 'wind_zone'])['windspeed_10m_max'].mean().unstack()

    def plot_combined_decomposition(data, title):
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        for zone in data.columns:
            print(f'Decomposing {title} for {zone}...')
            series = data[zone].dropna()
            if series.empty:
                continue
            res = seasonal_decompose(series, model='additive', period=12)
            axes[0].plot(res.observed, label=zone, alpha=0.7)
            axes[1].plot(res.trend, label=zone, alpha=0.7)
            axes[2].plot(res.seasonal, label=zone, alpha=0.7)
            axes[3].plot(res.resid, label=zone, alpha=0.7)
        axes[0].set_ylabel('Observed')
        axes[1].set_ylabel('Trend')
        axes[2].set_ylabel('Seasonal')
        axes[3].set_ylabel('Residual')
        axes[0].legend(loc='upper right', ncol=3, fontsize='small')
        plt.suptitle(f'{title} Decomposition (All Zones)', fontsize=16)
        plt.tight_layout()
        plt.show()
    plot_combined_decomposition(evap_monthly_1, 'Evaporation')
    plot_combined_decomposition(wind_monthly_1, 'Wind Speed')
    return


@app.cell
def _(WEATHER_1, adfuller, pd, warnings):
    from statsmodels.tsa.stattools import kpss
    warnings.filterwarnings('ignore')
    evap_monthly_2 = WEATHER_1.groupby([pd.Grouper(key='time', freq='M'), 'zone'])['et0_fao_evapotranspiration'].mean().unstack()
    wind_monthly_2 = WEATHER_1.groupby([pd.Grouper(key='time', freq='M'), 'wind_zone'])['windspeed_10m_max'].mean().unstack()

    def run_tests(series, name):
        res_adf = adfuller(series.dropna())
        res_kpss = kpss(series.dropna(), regression='c')
        return {'Variable': name, 'ADF_p': round(res_adf[1], 4), 'ADF_Stat': round(res_adf[0], 4), 'ADF_Result': 'Stationary' if res_adf[1] < 0.05 else 'Non-Stationary', 'KPSS_p': round(res_kpss[1], 4), 'KPSS_Result': 'Stationary' if res_kpss[1] > 0.05 else 'Non-Stationary'}
    results = []
    for zone in evap_monthly_2.columns:
        results.append(run_tests(evap_monthly_2[zone], f'Evap - {zone}'))  # ADF Test
    for zone in wind_monthly_2.columns:
        results.append(run_tests(wind_monthly_2[zone], f'Wind - {zone}'))  # KPSS Test
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    return


@app.cell
def _(WEATHER_1, pd, plt):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    evap_monthly_3 = WEATHER_1.groupby([pd.Grouper(key='time', freq='M'), 'zone'])['et0_fao_evapotranspiration'].mean().unstack()
    # Monthly Aggregation
    zones = evap_monthly_3.columns
    fig_6, axes_5 = plt.subplots(len(zones), 2, figsize=(15, 5 * len(zones)))
    for i_1, zone_1 in enumerate(zones):
        series = evap_monthly_3[zone_1].dropna()
        plot_acf(series, ax=axes_5[i_1, 0], lags=36)
        axes_5[i_1, 0].set_title(f'ACF: {zone_1} Evapotranspiration')
        plot_pacf(series, ax=axes_5[i_1, 1], lags=36, method='ywm')
        axes_5[i_1, 1].set_title(f'PACF: {zone_1} Evapotranspiration')
    plt.tight_layout()  # ACF Plot
    plt.show()  # PACF Plot
    return plot_acf, plot_pacf


@app.cell
def _(WEATHER_1, pd, plot_acf, plot_pacf, plt):
    # Monthly Aggregation
    evap_monthly_4 = WEATHER_1.groupby([pd.Grouper(key='time', freq='M'), 'wind_zone'])['windspeed_10m_max'].mean().unstack()
    zones_1 = evap_monthly_4.columns
    fig_7, axes_6 = plt.subplots(len(zones_1), 2, figsize=(15, 5 * len(zones_1)))
    for i_2, zone_2 in enumerate(zones_1):
        series_1 = evap_monthly_4[zone_2].dropna()
        plot_acf(series_1, ax=axes_6[i_2, 0], lags=36)
        axes_6[i_2, 0].set_title(f'ACF: {zone_2} Windspeed')
        plot_pacf(series_1, ax=axes_6[i_2, 1], lags=36, method='ywm')
        axes_6[i_2, 1].set_title(f'PACF: {zone_2} Windspeed')
    plt.tight_layout()  # ACF Plot
    plt.show()  # PACF Plot
    return


@app.cell
def _():
    #%pip install pmdarima
    return


@app.cell
def _(WEATHER_1, mean_squared_error, np, pd, plt, warnings):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_absolute_error
    import pmdarima as pm
    warnings.filterwarnings('ignore')

    def analyze_weather_variable(df, zone_name, target_col, zone_col):
        print(f'\n' + '=' * 60)
        print(f'ANALYZING: {zone_name} | VARIABLE: {target_col}')
        print('=' * 60)
        series = df[df[zone_col] == zone_name].groupby('time')[target_col].mean().resample('M').mean().dropna()
        if len(series) < 24:
            print(f'Skipping {zone_name}: Not enough data points for seasonal SARIMA.')
            return None
        train = series.iloc[:-12]  # Filter and Resample
        test = series.iloc[-12:]
        print(f'Optimizing SARIMA for {target_col}...')
        stepwise_model = pm.auto_arima(train, seasonal=True, m=12, d=1, D=1, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
        model = SARIMAX(train, order=stepwise_model.order, seasonal_order=stepwise_model.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        forecast_obj = results.get_forecast(steps=12)
        pred_mean = forecast_obj.predicted_mean  # Train-Test Split (Last 12 months for testing)
        conf_int = forecast_obj.conf_int()
        mae = mean_absolute_error(test, pred_mean)
        rmse = np.sqrt(mean_squared_error(test, pred_mean))
        print(f'Optimal Model: {stepwise_model.order}x{stepwise_model.seasonal_order}')  #Automated Parameter Selection
        print(f'MAE: {mae:.4f} | RMSE: {rmse:.4f}')
        plt.figure(figsize=(12, 5))
        plt.plot(train.index, train, label='Historical (Train)', color='#34495e', alpha=0.5)
        plt.plot(test.index, test, label='Actual (Test)', color='#27ae60', linewidth=2)
        plt.plot(pred_mean.index, pred_mean, label='SARIMA Forecast', color='#e67e22', linestyle='--')
        plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='#e67e22', alpha=0.1)
        plt.title(f'{target_col} Forecast: {zone_name}')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)  # Fit Final Model
        plt.show()
        return results
    evap_target = 'et0_fao_evapotranspiration'
    for zone_3 in WEATHER_1['zone'].unique():
        if pd.notna(zone_3):
            analyze_weather_variable(WEATHER_1, zone_3, evap_target, 'zone')
    wind_target = 'windspeed_10m_max'
    for zone_3 in WEATHER_1['wind_zone'].unique():  # Forecast & Evaluation
        if pd.notna(zone_3):
    # Analyze Evapotranspiration (using 'zone' column)
    # Analyze Wind Speed (using 'wind_zone' column)
            analyze_weather_variable(WEATHER_1, zone_3, wind_target, 'wind_zone')  # Visualization
    return (mean_absolute_error,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Weather effect on major hydro power generation in Sri Lanka
    """)
    return


@app.cell
def _(pd):
    file_id = '1j9BcmoAJUW90LC9FwaCfosbc2r17Ea7Y'
    download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    try:
        GENERATION = pd.read_csv(download_url)
        print("CSV imported successfully!")
        print(GENERATION.head())
    except Exception as e:
        print(f"An error occurred: {e}")
    return (GENERATION,)


@app.cell
def _(GENERATION, pd):
    # Convert to Datetime
    GENERATION['Date (GMT+5:30)'] = pd.to_datetime(GENERATION['Date (GMT+5:30)'])
    # Fill NaN values with 0
    GENERATION_1 = GENERATION.fillna(0)
    return (GENERATION_1,)


@app.cell
def _(
    GENERATION_1,
    RandomForestRegressor,
    WEATHER_1,
    mean_absolute_error,
    np,
    pd,
    plt,
    r2_score,
    zone_map,
):
    df_weather = WEATHER_1.copy()
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    if 'zone' not in df_weather.columns:
    # WEATHER DATA PREPARATION
        df_weather['zone'] = df_weather['city'].map(zone_map)
    weather_cols_1 = ['et0_fao_evapotranspiration', 'precipitation_sum']
    zonal_daily_1 = df_weather.groupby(['time', 'zone'])[weather_cols_1].mean(numeric_only=True).unstack()
    zonal_daily_1.columns = [f'{col[0]}_{col[1]}' for col in zonal_daily_1.columns]
    for zone_4 in ['Wet Zone', 'Intermediate Zone']:
        if f'precipitation_sum_{zone_4}' in zonal_daily_1.columns:
            zonal_daily_1[f'Rain_180_{zone_4}'] = zonal_daily_1[f'precipitation_sum_{zone_4}'].rolling(window=180).sum()
    # Feature Generation: Focus on Wet and Intermediate Zones
        if f'et0_fao_evapotranspiration_{zone_4}' in zonal_daily_1.columns:
            zonal_daily_1[f'ET0_90_{zone_4}'] = zonal_daily_1[f'et0_fao_evapotranspiration_{zone_4}'].rolling(window=90).mean()
    df_w = zonal_daily_1.resample('W-SUN').mean(numeric_only=True)
    gen_w = GENERATION_1.copy()
    # Cumulative Features (Reservoir)
    gen_w = gen_w.rename(columns={'Date (GMT+5:30)': 'time'})
    gen_w['time'] = pd.to_datetime(gen_w['time'])
    target_col = 'Major Hydro (Energy in GWh/day)'
    gen_w[target_col] = pd.to_numeric(gen_w[target_col], errors='coerce').fillna(0)
    gen_w = gen_w.set_index('time').resample('W-SUN').mean(numeric_only=True)
    df_final = pd.merge(gen_w[[target_col]], df_w, left_index=True, right_index=True, how='inner')
    # Weekly Resample (Sunday Anchor)
    feature_cols = [c for c in df_final.columns if 'Rain_180' in c or 'ET0_90' in c]
    df_final[feature_cols] = df_final[feature_cols].shift(1)
    # GENERATION DATA: ISOLATE MAJOR HYDRO
    df_final['day_sin'] = np.sin(2 * np.pi * df_final.index.dayofyear / 365.25)
    df_final = df_final.dropna()
    train_df_power = df_final[(df_final.index.year >= 2016) & (df_final.index.year <= 2018)]
    test_df_power = df_final[df_final.index.year == 2019]
    X_train_power, y_train_power = (train_df_power[feature_cols + ['day_sin']], train_df_power[target_col])
    X_test_power, y_test_power = (test_df_power[feature_cols + ['day_sin']], test_df_power[target_col])
    model_1 = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42)
    model_1.fit(X_train_power, y_train_power)
    # Weekly Resample
    y_pred = model_1.predict(X_test_power)
    r2 = r2_score(y_test_power, y_pred)
    # MERGE & LAG
    mae = mean_absolute_error(y_test_power, y_pred)
    mask = y_test_power > 0.1
    # PREVENT LEAKAGE: Shift features by 1 week
    safe_mape = np.mean(np.abs((y_test_power[mask] - y_pred[mask]) / y_test_power[mask])) * 100
    plt.figure(figsize=(15, 6))
    plt.plot(test_df_power.index, y_test_power, label='Actual Major Hydro', color='#34495e', linewidth=2)
    plt.plot(test_df_power.index, y_pred, label='Forecasted Major Hydro', color='#3498db', linestyle='--')
    plt.fill_between(test_df_power.index, y_test_power, y_pred, color='gray', alpha=0.2)
    # TRAIN (2016-2018) & TEST (2019)
    plt.title('Forecasting Major Hydro (Reservoir-Based) - 2019 Test Period', fontsize=14)
    plt.ylabel('GWh / Day')
    plt.legend()
    plt.show()
    print(f'--- Major Hydro Evaluation (2019) ---')
    print(f'R-squared: {r2:.3f}')
    # Random Forest
    print(f'MAE: {mae:.2f} GWh/day')
    # EVALUATION
    # VISUALIZATION
    print(f'MAPE: {safe_mape:.2f}%')
    return (
        X_test_power,
        X_train_power,
        df_weather,
        model_1,
        test_df_power,
        y_pred,
        y_test_power,
        y_train_power,
    )


@app.cell
def _(
    X_train_power,
    mean_absolute_error,
    model_1,
    r2_score,
    y_pred,
    y_test_power,
    y_train_power,
):
    y_train_pred = model_1.predict(X_train_power)
    train_r2 = r2_score(y_train_power, y_train_pred)
    test_r2 = r2_score(y_test_power, y_pred)
    train_mae = mean_absolute_error(y_train_power, y_train_pred)
    test_mae = mean_absolute_error(y_test_power, y_pred)
    print(f'--- Overfitting Check ---')
    print(f'Train R2: {train_r2:.3f} | Test R2: {test_r2:.3f}')
    print(f'Train MAE: {train_mae:.2f} | Test MAE: {test_mae:.2f}')
    if train_r2 > test_r2 + 0.15:
        print('DIAGNOSIS: Likely OVERFITTING (Model memorized the noise in 2016-2018 data)')
    elif train_r2 < 0.5 and test_r2 < 0.5:
        print('DIAGNOSIS: Likely UNDERFITTING (Model is too simple for this hydro data)')
    else:
        print('DIAGNOSIS: Good Fit')
    return


@app.cell
def _(
    RandomForestRegressor,
    X_test_power,
    X_train_power,
    mean_absolute_error,
    np,
    plt,
    r2_score,
    test_df_power,
    y_pred,
    y_test_power,
    y_train_power,
):
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    param_grid = {'max_depth': [3, 4, 5, 7], 'min_samples_leaf': [2, 5, 8, 12], 'max_features': ['sqrt', 1.0]}
    #SETUP GRID SEARCH
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(estimator=RandomForestRegressor(n_estimators=200, random_state=42), param_grid=param_grid, cv=tscv, scoring='r2', n_jobs=-1)  # Lower depths to reduce overfitting
    grid_search.fit(X_train_power, y_train_power)  # Higher numbers smooth out noise
    print(f'Best Params: {grid_search.best_params_}')
    print(f'Best Validation Score (Avg R2 during CV): {grid_search.best_score_:.3f}')
    best_model = grid_search.best_estimator_
    best_model.fit(X_train_power, y_train_power)
    y_train_pred_new = best_model.predict(X_train_power)
    y_test_pred_new = best_model.predict(X_test_power)
    train_r2_1 = r2_score(y_train_power, y_train_pred_new)
    train_mae_1 = mean_absolute_error(y_train_power, y_train_pred_new)
    test_r2_1 = r2_score(y_test_power, y_test_pred_new)
    test_mae_1 = mean_absolute_error(y_test_power, y_test_pred_new)
    mask_1 = y_test_power > 0.1
    safe_mape_1 = np.mean(np.abs((y_test_power[mask_1] - y_test_pred_new[mask_1]) / y_test_power[mask_1])) * 100
    print(f'\n--- Overfitting Check (Tuned Model) ---')
    #FIT AND FIND BEST PARAMS
    print(f'Train R2: {train_r2_1:.3f} | Test R2: {test_r2_1:.3f}')
    print(f'Train MAE: {train_mae_1:.2f} | Test MAE: {test_mae_1:.2f}')
    print(f'Test MAPE: {safe_mape_1:.2f}%')
    gap = train_r2_1 - test_r2_1
    if gap > 0.15:
    #RETRAIN BEST MODEL
        print(f'DIAGNOSIS: Still Overfitting (Gap: {gap:.2f}). Try increasing min_samples_leaf.')
    elif train_r2_1 < 0.5:
        print(f'DIAGNOSIS: Underfitting (Gap: {gap:.2f}). The model is too restricted.')
    #PREDICT BOTH TRAIN AND TEST (For Overfitting Check)
    else:  # <--- New Line
        print(f'DIAGNOSIS: Good Fit (Gap: {gap:.2f})')  # <--- Renamed to avoid confusion
    plt.figure(figsize=(15, 6))
    #CALCULATE METRICS
    # Train Metrics
    plt.plot(test_df_power.index, y_test_power, label='Actual Major Hydro', color='#34495e', linewidth=2)
    plt.plot(test_df_power.index, y_pred, label='Forecasted Major Hydro', color='#3498db', linestyle='--')
    plt.fill_between(test_df_power.index, y_test_power, y_pred, color='gray', alpha=0.2)
    # Test Metrics
    plt.title('Forecasting Major Hydro (Reservoir-Based) - 2019 Test Period', fontsize=14)
    plt.ylabel('GWh / Day')
    plt.legend()
    # MAPE (Using the NEW predictions)
    #PRINT DIAGNOSIS
    # VISUALIZATION
    plt.show()
    return (mask_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objective 4: Geographical Influences on Extreme Events in Climate Variables
    """)
    return


@app.cell
def _():
    from sklearn.ensemble import IsolationForest

    return (IsolationForest,)


@app.cell
def _(X_1, select, test_df, train_df):
    features_2 = ['temperature_2m_mean', 'shortwave_radiation_sum', 'precipitation_sum', 'windspeed_10m_max', 'elevation']
    iso_train_df = train_df >> select(X_1.time, X_1.city, *features_2)
    iso_test_df = test_df >> select(X_1.time, X_1.city, *features_2)
    return iso_test_df, iso_train_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Statistical Anomaly Detection (Z-Score)
    Points with a Z-score > 3 or < -3 are considered anomalies.
    """)
    return


app._unparsable_cell(
    r"""
    from dfply import *
    for var in ['temperature_2m_mean', 'shortwave_radiation_sum', 'precipitation_sum', 'windspeed_10m_max']:
        anomalies = (iso_train_df >>
                     mask(abs((X[var] - X[var].mean()) / X[var].std()) > 3) >>
                     arrange(X[var], ascending=False))

        print(f"Anomalies in {var}: {len(anomalies)}")
        if not anomalies.empty:
            print(anomalies[['time', 'city', var]] >> head(5))
        city_anomalies = anomalies['city'].value_counts().reset_index()
        city_anomalies.columns = ['city', 'anomaly_count']

        plt.figure(figsize=(5, 3))
        sns.barplot(x='anomaly_count', y='city', data=city_anomalies.head(), palette='viridis')
        plt.title(f'Count of Anomalies of {var} by City (Z-score)')
        plt.xlabel('Number of Anomalies')
        plt.ylabel('City')
        plt.tight_layout()
        plt.show()

        print("-" * 50)
    """,
    name="_"
)


@app.cell
def _(X_1, arrange, iso_train_df, mask_1):
    var_1 = 'temperature_2m_mean'
    mean_val = iso_train_df[var_1].mean()
    # Calculate the fixed 3rd sigma boundaries
    std_val = iso_train_df[var_1].std()
    upper_limit = mean_val + 3 * std_val
    lower_limit = mean_val - 3 * std_val
    anomalies = iso_train_df >> mask_1((X_1[var_1] > upper_limit) | (X_1[var_1] < lower_limit)) >> arrange(X_1.time, ascending=True)
    print(f'Lower Bound: {lower_limit:.2f}°C | Upper Bound: {upper_limit:.2f}°C')
    # Filter for points outside the 3-sigma band
    print(f'Total Anomalies found: {len(anomalies)}')
    return anomalies, lower_limit, upper_limit, var_1


@app.cell
def _(X_1, arrange, mask_1, plt):
    def plot_anomalies_for_city(city_name, var, iso_train_df, anomalies, lower_limit=None, upper_limit=None):
        plot_data = iso_train_df >> mask_1(X_1.city == city_name) >> arrange(X_1.time)
        city_anomalies_data = anomalies >> mask_1(X_1.city == city_name) >> arrange(X_1.time)
        plt.figure(figsize=(15, 6))
        plt.plot(plot_data.time, plot_data[var], label='Data', color='blue', alpha=0.3)
        plt.scatter(city_anomalies_data.time, city_anomalies_data[var], color='red', label='Anomaly', s=20, zorder=5)
        if lower_limit is not None or upper_limit is not None:
            if upper_limit is not None:
                plt.axhline(upper_limit, color='green', linestyle='--', alpha=0.7, label='Upper Cutoff (3σ)')
            if lower_limit is not None:
                plt.axhline(lower_limit, color='green', linestyle='--', alpha=0.7, label='Lower Cutoff (3σ)')
            if lower_limit is not None and upper_limit is not None:
                plt.fill_between(plot_data.time, lower_limit, upper_limit, color='green', alpha=0.05, label='Normal Range')
        plt.title(f'Time Series Anomalies & Z-Score Cutoffs in {city_name}: {var}')  # Plot the anomalies
        plt.xlabel('Time')
        plt.ylabel(var)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.show()  # Add the Cutoff Band  # Shade the "normal" region only if both limits exist to avoid plotting errors

    return (plot_anomalies_for_city,)


@app.cell
def _(
    anomalies,
    iso_train_df,
    lower_limit,
    plot_anomalies_for_city,
    upper_limit,
    var_1,
):
    plot_anomalies_for_city('Hatton', var_1, iso_train_df, anomalies, lower_limit, upper_limit)
    return


@app.cell
def _(X_1, arrange, iso_train_df, mask_1, var_1):
    median_val = iso_train_df['temperature_2m_mean'].median()
    mad_val = (iso_train_df[var_1] - median_val).abs().median()
    # Calculate the Median Absolute Deviation (MAD)
    consistency_constant = 1.4826
    upper_limit_1 = median_val + 3 * consistency_constant * mad_val
    # Define the threshold
    # We use 1.4826 as a scaling factor to make MAD comparable to
    # Standard Deviation for a normal distribution.
    lower_limit_1 = median_val - 3 * consistency_constant * mad_val
    anomalies_1 = iso_train_df >> mask_1((X_1[var_1] > upper_limit_1) | (X_1[var_1] < lower_limit_1)) >> arrange(X_1[var_1], ascending=False)
    print(f'Median: {median_val:.2f}°C')
    print(f'Lower Bound (Robust): {lower_limit_1:.2f}°C | Upper Bound (Robust): {upper_limit_1:.2f}°C')
    # Filter for points outside the robust band
    print(f'Total Anomalies found using Median: {len(anomalies_1)}')
    return anomalies_1, lower_limit_1, upper_limit_1


@app.cell
def _(
    anomalies_1,
    iso_train_df,
    lower_limit_1,
    plot_anomalies_for_city,
    upper_limit_1,
    var_1,
):
    plot_anomalies_for_city('Hatton', var_1, iso_train_df, anomalies_1, lower_limit_1, upper_limit_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Isolation Forest Approach
    """)
    return


@app.cell
def _(IsolationForest, iso_test_df, iso_train_df):
    iso_model = IsolationForest(contamination=0.01, random_state=42)
    iso_train_df['anomaly'] = iso_model.fit_predict(iso_train_df[
        ['temperature_2m_mean', 'shortwave_radiation_sum',
         'precipitation_sum', 'windspeed_10m_max']
        ])
    iso_test_df['anomaly'] = iso_model.predict(iso_test_df[
        ['temperature_2m_mean', 'shortwave_radiation_sum',
         'precipitation_sum', 'windspeed_10m_max']
        ])

    # -1 indicates anomaly, 1 indicates normal
    anomalies_iso = iso_train_df[iso_train_df['anomaly'] == -1]

    print(f"Total Anomalies Detected by Isolation Forest: {len(anomalies_iso)}")
    print("Sample Anomalies:")
    print(anomalies_iso.sort_values(by='temperature_2m_mean', ascending=False).head())
    return anomalies_iso, iso_model


@app.cell
def _(anomalies_iso, iso_train_df, plot_anomalies_for_city):
    for var_2 in ['temperature_2m_mean', 'shortwave_radiation_sum', 'precipitation_sum', 'windspeed_10m_max']:
        plot_anomalies_for_city('Hatton', var_2, iso_train_df, anomalies_iso)
    return


@app.cell
def _(anomalies_iso, plt, sns):
    # Count anomalies by city
    city_anomalies = anomalies_iso['city'].value_counts().reset_index()
    city_anomalies.columns = ['city', 'anomaly_count']

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='anomaly_count', y='city', data=city_anomalies, palette='viridis')
    plt.title('Count of Anomalies by City (Isolation Forest)')
    plt.xlabel('Number of Anomalies')
    plt.ylabel('City')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(iso_model, test_df, train_df):
    detect_vars = ['temperature_2m_mean', 'apparent_temperature_mean', 'shortwave_radiation_sum', 'precipitation_sum', 'windspeed_10m_max', 'weathercode', 'time', 'windgusts_10m_max', 'temperature_2m_min', 'apparent_temperature_min', 'temperature_2m_max', 'apparent_temperature_max', 'anomaly', 'rain_sum', 'snowfall_sum', 'is_anomaly', 'precip_intensity']
    train_df['anomaly'] = iso_model.fit_predict(train_df[['temperature_2m_mean', 'shortwave_radiation_sum', 'precipitation_sum', 'windspeed_10m_max']])
    train_df['is_anomaly'] = train_df['anomaly'].apply(lambda x: 1 if x == -1 else 0)
    test_df['anomaly'] = iso_model.predict(test_df[['temperature_2m_mean', 'shortwave_radiation_sum', 'precipitation_sum', 'windspeed_10m_max']])
    test_df['is_anomaly'] = test_df['anomaly'].apply(lambda x: 1 if x == -1 else 0)
    y_train_1 = train_df['is_anomaly']
    X_train_1 = train_df.drop([c for c in detect_vars if c in train_df.columns], axis=1)
    y_test_1 = test_df['is_anomaly']
    # Setting up train-test sets
    X_test_1 = test_df.drop([c for c in detect_vars if c in test_df.columns], axis=1)
    return X_test_1, X_train_1, y_test_1, y_train_1


@app.cell
def _(X_test_1, X_train_1, display, models, pd, y_test_1, y_train_1):
    from sklearn.metrics import confusion_matrix
    print('\n--- Test Set Evaluation ---')
    # Test Set Evaluation & Custom Table
    table_data = []
    for name, model_2 in models.items():
        try:
            model_2.fit(X_train_1, y_train_1)
            y_pred_1 = model_2.predict(X_test_1)
            y_prob = model_2.predict_proba(X_test_1)[:, 1]
            tn, fp, fn, tp = confusion_matrix(y_test_1, y_pred_1).ravel()
            table_data.append({'Model': name, 'TP (Caught)': tp, 'FN (Missed)': fn, 'FP (False Alarms)': fp, 'TN (Correct 0s)': tn})
        except Exception as e:
            print(f'Model {name} failed prediction: {e}')  # Calculate Confusion Matrix
    results_df_1 = pd.DataFrame(table_data)  # Confusion matrix returns [[TN, FP], [FN, TP]]
    if not results_df_1.empty:
        results_df_1 = results_df_1.set_index('Model')
    # Create DataFrame
        display(results_df_1.style.background_gradient(cmap='RdYlGn', subset=['TP (Caught)', 'TN (Correct 0s)']).background_gradient(cmap='RdYlGn_r', subset=['FN (Missed)', 'FP (False Alarms)']).format('{:,}'))  # Display formatted table style
    return (confusion_matrix,)


@app.cell
def _(models, pd, plt, sns):
    brf_pipeline = models['Balanced Random Forest']
    brf_model = brf_pipeline.named_steps['classifier']
    preprocessor = brf_pipeline.named_steps['preprocessor']
    try:
        feature_names = preprocessor.get_feature_names_out()
        importances_1 = pd.Series(brf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=importances_1.head(20).values, y=importances_1.head(20).index, palette='viridis')
        plt.title('Top 20 Feature Importance (Balanced Random Forest)')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f'Could not plot feature importance: {e}')
    return


@app.cell
def _(
    BalancedRandomForestClassifier,
    ColumnTransformer,
    EasyEnsembleClassifier,
    GradientBoostingClassifier,
    ImbPipeline,
    LogisticRegression,
    RandomForestClassifier,
    RandomUnderSampler,
    SMOTE,
    SMOTEENN,
    StandardScaler,
    StratifiedKFold,
    XGBClassifier,
    X_test_1,
    X_train_1,
    confusion_matrix,
    display,
    pd,
    select,
    test_df,
    train_df,
    y_test_1,
    y_train_1,
):
    X_train_if = train_df >> select(['et0_fao_evapotranspiration', 'precipitation_hours', 'winddirection_10m_dominant'])
    X_test_if = test_df >> select(['et0_fao_evapotranspiration', 'precipitation_hours', 'winddirection_10m_dominant'])
    numeric_features = [str(c) for c in X_train_if.select_dtypes(include=['float64', 'int64']).columns.tolist()]
    numeric_transformer = StandardScaler()
    preprocessor_1 = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)], verbose_feature_names_out=False)
    models = {}
    models['LogReg'] = ImbPipeline(steps=[('preprocessor', preprocessor_1), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
    models['RandomForest'] = ImbPipeline(steps=[('preprocessor', preprocessor_1), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    models['GradientBoosting'] = ImbPipeline(steps=[('preprocessor', preprocessor_1), ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))])
    models['XGBoost'] = ImbPipeline(steps=[('preprocessor', preprocessor_1), ('classifier', XGBClassifier(n_estimators=100, random_state=42))])
    models['LogReg (Balanced)'] = ImbPipeline(steps=[('preprocessor', preprocessor_1), ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])
    models['Elastic Net (Balanced)'] = ImbPipeline(steps=[('preprocessor', preprocessor_1), ('classifier', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000, class_weight='balanced', random_state=42))])
    models['SMOTE + LogReg'] = ImbPipeline(steps=[('preprocessor', preprocessor_1), ('resampler', SMOTE(random_state=42)), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
    # Define Expanded Model Suite
    models['SMOTEENN + LogReg'] = ImbPipeline(steps=[('preprocessor', preprocessor_1), ('resampler', SMOTEENN(random_state=42)), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
    models['RandomUnderSampler + LogReg'] = ImbPipeline(steps=[('preprocessor', preprocessor_1), ('resampler', RandomUnderSampler(random_state=42)), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
    models['Balanced Random Forest'] = ImbPipeline(steps=[('preprocessor', preprocessor_1), ('classifier', BalancedRandomForestClassifier(n_estimators=100, random_state=42))])
    models['Easy Ensemble'] = ImbPipeline(steps=[('preprocessor', preprocessor_1), ('classifier', EasyEnsembleClassifier(n_estimators=10, random_state=42))])
    print('--- Cross-Validation Scores (ROC-AUC) ---')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print('\n--- Test Set Evaluation ---')
    table_data_1 = []
    for name_1, model_3 in models.items():
        try:
            model_3.fit(X_train_1, y_train_1)
            y_pred_2 = model_3.predict(X_test_1)
            y_prob_1 = model_3.predict_proba(X_test_1)[:, 1]
            tn_1, fp_1, fn_1, tp_1 = confusion_matrix(y_test_1, y_pred_2).ravel()
            table_data_1.append({'Model': name_1, 'TP (Caught)': tp_1, 'FN (Missed)': fn_1, 'FP (False Alarms)': fp_1, 'TN (Correct 0s)': tn_1})
        except Exception as e:
            print(f'Model {name_1} failed prediction: {e}')
    results_df_2 = pd.DataFrame(table_data_1)
    if not results_df_2.empty:
        results_df_2 = results_df_2.set_index('Model')
    # --- Imbalance Strategies (Resampling) ---
    # --- Ensemble Methods for Imbalance ---
    # Evaluation
    # 6. Test Set Evaluation & Custom Table
    # Create DataFrame
        display(results_df_2.style.background_gradient(cmap='RdYlGn', subset=['TP (Caught)', 'TN (Correct 0s)']).background_gradient(cmap='RdYlGn_r', subset=['FN (Missed)', 'FP (False Alarms)']).format('{:,}'))  # Calculate Confusion Matrix  # Confusion matrix returns [[TN, FP], [FN, TP]]  # Display formatted table style
    return (models,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hyperparameter Tuning
    """)
    return


@app.cell
def _(StratifiedKFold, X_train_1, models, np, y_train_1):
    from sklearn.model_selection import RandomizedSearchCV
    print('--- Hyperparameter Tuning: Balanced Random Forest ---')
    param_dist_brf = {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [None, 10, 20, 30], 'classifier__min_samples_split': [2, 5, 10], 'classifier__min_samples_leaf': [1, 2, 4]}
    brf_tune = RandomizedSearchCV(models['Balanced Random Forest'], param_distributions=param_dist_brf, n_iter=10, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='roc_auc', n_jobs=-1, random_state=42)
    # 1. Define Parameter Grid
    brf_tune.fit(X_train_1, y_train_1)
    print(f'Best BRF Score: {brf_tune.best_score_:.4f}')
    print(f'Best BRF Params: {brf_tune.best_params_}')
    print('\n--- Hyperparameter Tuning: LogReg (Balanced) ---')
    param_dist_lr = {'classifier__C': np.logspace(-4, 4, 20), 'classifier__solver': ['liblinear', 'saga']}
    lr_tune = RandomizedSearchCV(models['LogReg (Balanced)'], param_distributions=param_dist_lr, n_iter=10, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='roc_auc', n_jobs=-1, random_state=42)
    lr_tune.fit(X_train_1, y_train_1)
    # 2. Initialize RandomizedSearch
    print(f'Best LogReg Score: {lr_tune.best_score_:.4f}')
    print(f'Best LogReg Params: {lr_tune.best_params_}')
    models['Balanced RF (Tuned)'] = brf_tune.best_estimator_
    # 3. Fit
    # Update models dictionary with best estimators
    models['LogReg Balanced (Tuned)'] = lr_tune.best_estimator_
    return


@app.cell
def _(X_test_1, confusion_matrix, display, models, pd, y_test_1):
    table_data_tuned = []
    tuned_model_names = ['Balanced RF (Tuned)', 'LogReg Balanced (Tuned)']
    for name_2 in tuned_model_names:
        model_4 = models[name_2]
        y_pred_3 = model_4.predict(X_test_1)
        tn_2, fp_2, fn_2, tp_2 = confusion_matrix(y_test_1, y_pred_3).ravel()
        table_data_tuned.append({'Model': name_2, 'TP (Caught)': tp_2, 'FN (Missed)': fn_2, 'FP (False Alarms)': fp_2, 'TN (Correct 0s)': tn_2})
    results_tuned_df = pd.DataFrame(table_data_tuned).set_index('Model')
    display(results_tuned_df.style.background_gradient(cmap='RdYlGn', subset=['TP (Caught)', 'TN (Correct 0s)']).background_gradient(cmap='RdYlGn_r', subset=['FN (Missed)', 'FP (False Alarms)']).format('{:,}'))
    return


if __name__ == "__main__":
    app.run()
