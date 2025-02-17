import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


env_df = pd.read_csv('NEW_ENVIRONMENT_DATASET.csv')
build_df = pd.read_csv('NEW_BUILDING_DATASET.csv')

env_df.drop(['Unnamed: 0'], axis=1, inplace=True)
build_df.drop(['Unnamed: 0'], axis=1, inplace=True)


# Air quality analysis
def air_quality_analysis():
    # Find the unique sensor location
    unique_sensor_location = env_df['SensorLocation'].unique().tolist()

    # For each of the region, find the average of each pollutant
    unique_pollutants = [i for i in env_df.keys().tolist() if 'pollutant_' in i.lower()]
    average_pollutants_per_location = []

    for sensor_loc in unique_sensor_location:
        temp_df = env_df[env_df['SensorLocation'] == sensor_loc]
        temp_arr_avg_pollutants = []

        for pollutants in unique_pollutants:
            avg_pollutants = temp_df[pollutants].mean()
            temp_arr_avg_pollutants.append(avg_pollutants)

        average_pollutants_per_location.append(temp_arr_avg_pollutants)

    # Use a bar graph with x-axis as the location and the y as the pollutants --> grouped bar graphs
    y_axis_arr = ['PM2.5', 'PM10', 'O3', 'NO2', 'CO', 'SO2']
    averages = [sum(location_data) / len(location_data) for location_data in average_pollutants_per_location]
    print(averages)

    # Parameters
    num_locations = len(unique_sensor_location)
    num_pollutants = len(y_axis_arr)
    bar_width = 1 / (num_pollutants + 1)  # Width of each bar
    bar_spacing = bar_width  # Spacing between bars within a group

    # Set the figure size
    plt.figure(figsize=(12, 6))

    # Plot each pollutant
    for i, pollutant in enumerate(y_axis_arr):
        # Calculate the bar positions for this pollutant
        bar_positions = [j + i * bar_spacing for j in range(num_locations)]
        # Extract pollutant values for all locations
        values = [average_pollutants_per_location[j][i] for j in range(num_locations)]
        # Plot the bars
        plt.bar(bar_positions, values, width=bar_width, label=pollutant)

    # Calculate positions for averages line graph
    average_positions = [j + (num_pollutants - 1) * bar_spacing / 2 for j in range(num_locations)]

    # Overlay the line graph for averages
    plt.plot(average_positions, averages, marker='o', color='red', label='Average Pollutant Level', linewidth=2)

    # Customize the graph
    plt.xlabel('Sensor Location', fontsize=12)
    plt.ylabel('Pollutant Level (µg/m³)', fontsize=12)
    plt.title('Pollutant Levels by Location', fontsize=14)

    # Add x-axis ticks and labels
    group_positions = [j + (num_pollutants - 1) * bar_spacing / 2 for j in range(num_locations)]
    plt.xticks(group_positions, unique_sensor_location, fontsize=10)

    # Add legend
    plt.legend(title='Pollutants', fontsize=10)

    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the graph
    plt.tight_layout()
    plt.show()


def get_coor_report():
    temp_df = env_df.drop(['SensorID', 'SensorLocation', 'EnergySavingTechnology', 'RetrofitData', 'Country'], axis=1)
    print(temp_df.corr(method='pearson').to_string())


def analysis_of_PM_gases():
    # Extract the PM gases and the correlated features
    temp_df = env_df[['Pollutant_PM2.5_µg/m³', 'Pollutant_PM10_µg/m³', 'UrbanVegetationArea_m2', 'Humidity_%', 'PopulationDensity_people/km²', 'AQI_Index']]

    # Combined scatterplot for PM2.5 and PM10 against all other factors
    factors = ['UrbanVegetationArea_m2', 'Humidity_%', 'PopulationDensity_people/km²', 'AQI_Index']

    for factor in factors:
        plt.figure(figsize=(8, 6))

        # PM2.5
        sns.scatterplot(data=temp_df, x=factor, y='Pollutant_PM2.5_µg/m³', label='PM2.5', color='blue')

        # PM10
        sns.scatterplot(data=temp_df, x=factor, y='Pollutant_PM10_µg/m³', label='PM10', color='orange')

        # Adding titles and labels
        plt.title(f'{factor} vs PM2.5 and PM10')
        plt.xlabel(factor)
        plt.ylabel('Pollutant Concentration (µg/m³)')
        plt.legend()
        plt.show()


def analysis_of_other_gas_pollutants():
    # Extract the other gas pollutants and the other factors correlated with it
    temp_df = env_df[['SensorLocation', 'Pollutant_O3_ppb', 'Pollutant_NO2_ppb', 'Pollutant_CO_ppm', 'Pollutant_SO2_ppb', 'UrbanVegetationArea_m2', 'PopulationDensity_people/km²', 'AQI_Index']]
    all_df = temp_df.drop(['SensorLocation'], axis=1)

    # plot the graph for the public features
    factors = ['UrbanVegetationArea_m2', 'PopulationDensity_people/km²', 'AQI_Index']

    for factor in factors:
        plt.figure(figsize=(12, 8))

        # O3
        sns.scatterplot(data=all_df, x=factor, y="Pollutant_O3_ppb", label='O3', color='red')

        # NO2
        sns.scatterplot(data=all_df, x=factor, y="Pollutant_NO2_ppb", label='NO2', color='green')

        # CO
        sns.scatterplot(data=all_df, x=factor, y="Pollutant_CO_ppm", label='CO', color='blue')

        # SO2
        sns.scatterplot(data=all_df, x=factor, y="Pollutant_SO2_ppb", label='SO2', color='yellow')

        # Add titles and labels
        plt.title(f"{factor} VS O3, NO2, CO and SO2")
        plt.xlabel(factor)
        plt.ylabel('Pollutant Concentration (µg/m³)')
        plt.legend()
        plt.show()


def analysis_of_other_gas_on_location():
    temp_df = env_df[['SensorLocation', 'Pollutant_O3_ppb', 'Pollutant_NO2_ppb', 'Pollutant_CO_ppm', 'Pollutant_SO2_ppb']]

    # List of pollutants to plot
    pollutants = ['Pollutant_NO2_ppb', 'Pollutant_CO_ppm', 'Pollutant_SO2_ppb']
    pollutant_labels = ['NO2 (ppb)', 'CO (ppm)', 'SO2 (ppb)']
    colors = ['green', 'blue', 'yellow']

    # Create the trend line plot
    plt.figure(figsize=(12, 8))

    for i, pollutant in enumerate(pollutants):
        sns.lineplot(
            data=temp_df,
            x='SensorLocation',
            y=pollutant,
            label=pollutant_labels[i],
            color=colors[i],
            marker='o'
        )

    # Add titles and labels
    plt.title('Average Pollutant Concentrations by Sensor Location', fontsize=14)
    plt.xlabel('Sensor Location', fontsize=12)
    plt.ylabel('Pollutant Concentration', fontsize=12)
    plt.legend(title='Pollutants', fontsize=10)
    plt.grid(visible=True, linestyle='--', alpha=0.7)

    # Show plot
    plt.tight_layout()
    plt.show()


def analysis_of_environmental_impacts():
    # Analyze the impacts of urban vegetation and green space index as well as population density against AQI
    temp_df = env_df[['UrbanVegetationArea_m2', 'GreenSpaceIndex_%', 'PopulationDensity_people/km²', 'AQI_Index']]
    print(temp_df.corr().to_string())

    # Set up the figure for multiple subplots
    plt.figure(figsize=(15, 5))

    # Scatter plot for Urban Vegetation Area vs AQI Index
    sns.scatterplot(x="UrbanVegetationArea_m2", y="AQI_Index", data=temp_df, color="blue", s=100)
    plt.title("Urban Vegetation Area vs AQI Index")
    plt.xlabel("Urban Vegetation Area (m²)")
    plt.ylabel("AQI Index")
    plt.tight_layout()
    plt.show()

    # Scatter plot for Green Space Index vs AQI Index
    sns.scatterplot(x="GreenSpaceIndex_%", y="AQI_Index", data=temp_df, color="green", s=100)
    plt.title("Green Space Index vs AQI Index")
    plt.xlabel("Green Space Index (%)")
    plt.ylabel("AQI Index")
    plt.tight_layout()
    plt.show()

    # Scatter plot for Population Density vs AQI Index
    sns.scatterplot(x="PopulationDensity_people/km²", y="AQI_Index", data=temp_df, color="red", s=100)
    plt.title("Population Density vs AQI Index")
    plt.xlabel("Population Density (people/km²)")
    plt.ylabel("AQI Index")
    plt.tight_layout()
    plt.show()


def analysis_of_energy_efficiency():
    # Analyze the annual savings towards energy saving technologies
    # Analyze also renewable energy percentage towards different regions and energy consumption
    # Analyze also how each country varies in energy consumption as well as the average renewable energy percentage
    temp_df = env_df[['AnnualEnergySavings_%', 'EnergySavingTechnology', 'RenewableEnergyPercentage_%', 'AnnualEnergyConsumption_kWh', 'SensorLocation', 'Country', 'CountryCode', 'EnergySavingTechnologyCode', 'SensorLocationCode']]
    print(temp_df.drop(['Country', 'EnergySavingTechnology', 'SensorLocation'], axis=1).corr().to_string())

    # First analysis
    df2 = temp_df[['AnnualEnergyConsumption_kWh', 'RenewableEnergyPercentage_%', 'SensorLocation']]
    # Plot 1: Average Renewable Energy Percentage vs Sensor Location
    # Grouping data by 'SensorLocation' and calculating the mean of 'RenewableEnergyPercentage_%'
    average_renewable_energy = df2.groupby("SensorLocation")["RenewableEnergyPercentage_%"].mean().reset_index()
    print(average_renewable_energy)

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x="SensorLocation",
        y="RenewableEnergyPercentage_%",
        data=average_renewable_energy,
        color="blue"
    )
    plt.title("Average Renewable Energy Percentage Across Different Regions", fontsize=14)
    plt.xlabel("Sensor Location")
    plt.ylabel("Average Renewable Energy Percentage (%)")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True, linestyle="--", alpha=0.5)  # Add a grid for better visualization
    plt.show()

    # Second Analysis
    df4 = temp_df[['RenewableEnergyPercentage_%', 'AnnualEnergySavings_%']]
    # Create scatterplot
    plt.figure(figsize=(8, 6))
    plt.scatter(df4['RenewableEnergyPercentage_%'], df4['AnnualEnergySavings_%'], color='blue', alpha=0.7)

    # Add titles and labels
    plt.title("Scatterplot of Renewable Energy Percentage vs. Annual Energy Savings", fontsize=14)
    plt.xlabel("Renewable Energy Percentage (%)", fontsize=12)
    plt.ylabel("Annual Energy Savings (%)", fontsize=12)

    # Show grid and plot
    plt.grid(alpha=0.3)
    plt.show()


# Analyze the build df
def preanalysis():
    # For each renewable types, print out the renewable capacity and the renewable contribution and print out both averages
    temp_df = build_df[['RenewableType', 'RenewableContributionPercentage', 'RenewableCapacity_kWh']]

    # Group by RenewableType and calculate the mean for both columns
    avg_data = temp_df.groupby('RenewableType').agg({
        'RenewableContributionPercentage': 'mean',
        'RenewableCapacity_kWh': 'mean'
    }).reset_index()
    print(avg_data.to_string())

    # Creating the first line graph for Renewable Contribution Percentage
    plt.figure(figsize=(10, 6))
    plt.plot(avg_data['RenewableType'], avg_data['RenewableContributionPercentage'],
             label='Avg Renewable Contribution (%)', marker='o')
    plt.xlabel('Renewable Type')
    plt.ylabel('Average Renewable Contribution (%)')
    plt.title('Average Renewable Contribution by Renewable Type')
    plt.grid(True)
    plt.legend()
    # plt.show()

    # Creating the second line graph for Renewable Capacity (kWh)
    plt.figure(figsize=(10, 6))
    plt.plot(avg_data['RenewableType'], avg_data['RenewableCapacity_kWh'], label='Avg Renewable Capacity (kWh)',
             marker='o')
    plt.xlabel('Renewable Type')
    plt.ylabel('Average Renewable Capacity (kWh)')
    plt.title('Average Renewable Capacity by Renewable Type')
    plt.grid(True)
    plt.legend()
    plt.show()


def analysis_of_monthly_electricity_consumption():
    # Analyze the factors affecting the monthly electricity consumption
    temp_df = build_df[['BuildingType', 'YearBuilt', 'MonthlyElectricityConsumption_kWh', 'PeakUsageTime_Hour', 'RenewableCapacity_kWh', 'RenewableType', 'RenewableContributionPercentage', 'EnergySource', 'EnergyEfficiency_kWh_per_m2', 'WeatherData_Temperature_C', 'WeatherData_SolarIntensity_Hours', 'WeatherData_WindSpeed_km_h', 'RenewableTypeCode', 'BuildingTypeCode', 'EnergySourceCode']]
    print(temp_df.drop(['BuildingType', 'RenewableType', 'EnergySource'], axis=1).corr(method='pearson').to_string())

    analysis_df = temp_df[['MonthlyElectricityConsumption_kWh', 'EnergyEfficiency_kWh_per_m2', 'BuildingType']]
    # Scatter plot to visualize the correlation
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=analysis_df, x='EnergyEfficiency_kWh_per_m2', y='MonthlyElectricityConsumption_kWh', hue='BuildingType',
                    style='BuildingType', s=100)
    plt.title('Correlation between Monthly Electricity Consumption and Energy Efficiency by Building Type')
    plt.xlabel('Energy Efficiency (kWh/m²)')
    plt.ylabel('Monthly Electricity Consumption (kWh)')
    plt.legend(title='Building Type')
    plt.grid(True)
    plt.show()


def analysis_of_renewables():
    # Analyze the factors affecting the renewable capacity and the renewable contribution
    temp_df = build_df[['BuildingType', 'YearBuilt', 'MonthlyElectricityConsumption_kWh', 'PeakUsageTime_Hour', 'RenewableCapacity_kWh', 'RenewableType', 'RenewableContributionPercentage', 'EnergySource', 'EnergyEfficiency_kWh_per_m2', 'WeatherData_Temperature_C', 'WeatherData_SolarIntensity_Hours', 'WeatherData_WindSpeed_km_h', 'RenewableTypeCode', 'BuildingTypeCode', 'EnergySourceCode']]
    print(temp_df.drop(['BuildingType', 'RenewableType', 'EnergySource'], axis=1).corr(method='pearson').to_string())

    analysis_df = temp_df[['RenewableCapacity_kWh', 'RenewableContributionPercentage', 'RenewableType']]

    # Scatter plot to visualize the correlation
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=analysis_df, x='RenewableContributionPercentage', y='RenewableCapacity_kWh', hue='RenewableType',
                    style='RenewableType', s=100)
    plt.title('Correlation between Renewable Capacity and Renewable Contribution by Renewable Type')
    plt.xlabel('Renewable Contribution Percentage (%)')
    plt.ylabel('Renewable Capacity (kWh)')
    plt.legend(title='Renewable Type')
    plt.grid(True)
    plt.show()


def analysis_of_each_building_type():
    # Analyze, for each building, the average of monthly electricity consumption, renewable contribution, energy efficiency regardless of time series
    temp_df = build_df[['BuildingType', 'MonthlyElectricityConsumption_kWh', 'RenewableContributionPercentage', 'EnergyEfficiency_kWh_per_m2']]

    # Group by BuildingType and calculate the average (not necessary here since the data is already per building type)
    avg_data = temp_df.groupby('BuildingType').mean()

    # First Line Graph: Monthly Electricity Consumption
    plt.figure(figsize=(12, 6))
    plt.plot(avg_data.index, avg_data['MonthlyElectricityConsumption_kWh'], marker='o',
             label='Avg Monthly Electricity (kWh)', color='blue')
    plt.xlabel('Building Type')
    plt.ylabel('Average Monthly Electricity (kWh)')
    plt.title('Average Monthly Electricity Consumption by Building Type')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analysis_of_electricity_to_each_renewable_type():
    # Analyze, for each renewable type, the effects of factors WeatherData_Temperature_C, WeatherData_SolarIntensity_Hours, WeatherData_WindSpeed_km_h and thus the average monthly electricity consumption for each renewable types
    temp_df = build_df[['MonthlyElectricityConsumption_kWh', 'RenewableType', 'WeatherData_Temperature_C', 'WeatherData_SolarIntensity_Hours', 'WeatherData_WindSpeed_km_h']]

    # Calculate averages grouped by Renewable Type
    grouped_avg = temp_df.groupby('RenewableType').mean()

    # First Graph: Average Monthly Electricity Consumption by Renewable Type
    plt.figure(figsize=(10, 5))
    plt.plot(grouped_avg.index, grouped_avg['MonthlyElectricityConsumption_kWh'], marker='o', color='blue',
             label='Avg Monthly Electricity (kWh)')
    plt.title('Average Monthly Electricity Consumption by Renewable Type')
    plt.xlabel('Renewable Type')
    plt.ylabel('Average Monthly Electricity (kWh)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Leverage machine learning models to predict the future analysis
def AQI_index_prediction():
    # Use machine learning from the combined df to predict the future AQI index based on influential factors
    temp_df = env_df[['Pollutant_PM2.5_µg/m³', 'Pollutant_PM10_µg/m³', 'Pollutant_O3_ppb', 'Pollutant_NO2_ppb', 'Pollutant_CO_ppm', 'Pollutant_SO2_ppb', 'AQI_Index']]

    # Features and target
    X = temp_df[['Pollutant_PM2.5_µg/m³', 'Pollutant_PM10_µg/m³', 'Pollutant_O3_ppb',
            'Pollutant_NO2_ppb', 'Pollutant_CO_ppm', 'Pollutant_SO2_ppb']]
    y = temp_df['AQI_Index']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Output coefficients for analysis
    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })
    print("\nFeature Coefficients:")
    print(coefficients)

    # Predictions for plotting regression lines
    predictions = {}
    for feature in X.columns:
        # Fit model for the individual feature
        feature_model = LinearRegression()
        feature_data = X[[feature]]
        feature_model.fit(feature_data, y)
        predictions[feature] = feature_model.predict(feature_data)

    # Plot scatterplots with regression lines
    for feature in X.columns:
        plt.figure(figsize=(8, 5))
        plt.scatter(X[feature], y, color="blue", alpha=0.7, label="Actual Data")
        plt.plot(X[feature], predictions[feature], color="red", label="Prediction line")
        plt.title(f"Scatterplot of {feature} vs AQI Index", fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel("AQI Index", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

if __name__ == '__main__':
    air_quality_analysis()
    get_coor_report()
    analysis_of_PM_gases()
    analysis_of_other_gas_pollutants()
    analysis_of_other_gas_on_location()
    analysis_of_environmental_impacts()
    analysis_of_energy_efficiency()
    preanalysis()
    analysis_of_monthly_electricity_consumption()
    analysis_of_renewables()
    analysis_of_each_building_type()
    analysis_of_electricity_to_each_renewable_type()
    AQI_index_prediction()