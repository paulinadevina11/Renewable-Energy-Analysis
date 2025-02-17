import pandas as pd
import warnings
import sklearn.preprocessing as pp

warnings.filterwarnings('ignore')

env_df = pd.read_csv('environmental_dataset.csv')
build_df = pd.read_csv('building_dataset.csv')

# Know how much is null and take action on N/A datas
description_report_environment = env_df.describe()
null_sum = env_df.isnull().sum()

# It is known that 
# 2521 data of UrbanVegetationArea_m2
# 1094 data of EnergySavingTechnology
# 1024 data of PopulationDensity_people/km²
# 749 data of RenewableEnergyPercentage_%
# is all None

description_report_building = build_df.describe()
null_report = build_df.isnull().sum()

# It is known that
# 7219 data of YearBuilt
# 568 data of RenewableType
# 654 data of RenewableContributionPercentage
# 317 data of WeatherData_WindSpeed_km_h
# is all None

def env_df_cleaned():
    # For UrbanVegetationArea_m2, PopulationDensity_people/km² and RenewableEnergyPercentage_%, find the average and impute all that is None with the average
    average_urban_vegetation = env_df['UrbanVegetationArea_m2'].mean()
    env_df['UrbanVegetationArea_m2'].fillna(average_urban_vegetation, inplace=True)

    average_population_density = env_df['PopulationDensity_people/km²'].mean()
    env_df['PopulationDensity_people/km²'].fillna(average_population_density, inplace=True)

    average_RE_percent = env_df['RenewableEnergyPercentage_%'].mean()
    env_df['RenewableEnergyPercentage_%'].fillna(average_RE_percent, inplace=True)

    env_df.dropna(inplace=True)

    # Now encode the energy savings
    energy_saving_encode = pp.LabelEncoder()
    energy_saving_encode.fit(env_df['EnergySavingTechnology'])
    env_df['EnergySavingTechnologyCode'] = energy_saving_encode.transform(env_df['EnergySavingTechnology'])

    # Encode the SensorLocation
    sensor_locations_encode = pp.LabelEncoder()
    sensor_locations_encode.fit(env_df['SensorLocation'])
    env_df['SensorLocationCode'] = sensor_locations_encode.transform(env_df['SensorLocation'])

    # Encode the RetrofitData
    retrofit_encode = pp.LabelEncoder()
    retrofit_encode.fit(env_df['RetrofitData'])
    env_df['RetrofitDataCode'] = retrofit_encode.transform(env_df['RetrofitData'])

    # Encode the Country
    country_encode = pp.LabelEncoder()
    country_encode.fit(env_df['Country'])
    env_df['CountryCode'] = country_encode.transform(env_df['Country'])

    return env_df

def build_df_cleaned():
    # For the YearBuilt, we will assume it is now that is built by using interpolation. In case there are N/A values, delete them
    build_df['YearBuilt'] = build_df['YearBuilt'].interpolate(method='linear')
    build_df['YearBuilt'].fillna(2024, inplace=True)

    # For the RenewableContributionPercentage and WeatherData_WindSpeed_km_h, use average
    average_RC_percent = build_df['RenewableContributionPercentage'].mean()
    build_df['RenewableContributionPercentage'].fillna(average_RC_percent, inplace=True)

    average_windspeed = build_df['WeatherData_WindSpeed_km_h'].mean()
    build_df['WeatherData_WindSpeed_km_h'].fillna(average_windspeed, inplace=True)

    build_df.dropna(inplace=True)

    renewable_type_code = pp.LabelEncoder()
    renewable_type_code.fit(build_df['RenewableType'])
    build_df['RenewableTypeCode'] = renewable_type_code.transform(build_df['RenewableType'])


    building_type_code = pp.LabelEncoder()
    building_type_code.fit(build_df['BuildingType'])
    build_df['BuildingTypeCode'] = building_type_code.transform(build_df['BuildingType'])

    EnergySource_code = pp.LabelEncoder()
    EnergySource_code.fit(build_df['EnergySource'])
    build_df['EnergySourceCode'] = EnergySource_code.transform(build_df['EnergySource'])

    return build_df

if __name__ == '__main__':
    env_df_new = env_df_cleaned()
    build_df_new = build_df_cleaned()

    # env_df_new.to_csv('NEW_ENVIRONMENT_DATASET.csv')
    # build_df_new.to_csv('NEW_BUILDING_DATASET.csv')
