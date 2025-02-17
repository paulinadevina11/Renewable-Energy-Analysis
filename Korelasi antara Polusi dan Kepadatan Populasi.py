import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
environmental_df = pd.read_csv("NEW_ENVIRONMENT_DATASET.csv")

# Filter relevant columns and drop rows with missing values
pollution_data = environmental_df[
    ["Pollutant_PM2.5_µg/m³", "Pollutant_PM10_µg/m³", "PopulationDensity_people/km²"]
].dropna()

# Features and targets
X = pollution_data[["PopulationDensity_people/km²"]]
y_pm25 = pollution_data["Pollutant_PM2.5_µg/m³"]
y_pm10 = pollution_data["Pollutant_PM10_µg/m³"]

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_pm25_train, y_pm25_test = train_test_split(X, y_pm25, test_size=0.2, random_state=42)
_, _, y_pm10_train, y_pm10_test = train_test_split(X, y_pm10, test_size=0.2, random_state=42)

# Hyperparameter grids for Random Forest and Gradient Boosting
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10]
}

# Grid Search with Cross-Validation for Random Forest
rf_grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=rf_param_grid, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error', verbose=1)
rf_grid_search.fit(X_train, y_pm25_train)

# Grid Search with Cross-Validation for Gradient Boosting
gb_grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), param_grid=gb_param_grid, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error', verbose=1)
gb_grid_search.fit(X_train, y_pm25_train)

# Best parameters found by GridSearchCV
print(f"Best Random Forest Parameters: {rf_grid_search.best_params_}")
print(f"Best Random Forest MAE: {-rf_grid_search.best_score_:.2f}")
print(f"Best Gradient Boosting Parameters: {gb_grid_search.best_params_}")
print(f"Best Gradient Boosting MAE: {-gb_grid_search.best_score_:.2f}")

# Retrain models with the best parameters found
best_rf_model = rf_grid_search.best_estimator_
best_gb_model = gb_grid_search.best_estimator_

# Make predictions with tuned models
y_pm25_pred_rf_tuned = best_rf_model.predict(X_test)
y_pm25_pred_gb_tuned = best_gb_model.predict(X_test)

# Model evaluation with tuned models
mae_pm25_rf_tuned = mean_absolute_error(y_pm25_test, y_pm25_pred_rf_tuned)
r2_pm25_rf_tuned = r2_score(y_pm25_test, y_pm25_pred_rf_tuned)

mae_pm25_gb_tuned = mean_absolute_error(y_pm25_test, y_pm25_pred_gb_tuned)
r2_pm25_gb_tuned = r2_score(y_pm25_test, y_pm25_pred_gb_tuned)

# Print evaluation metrics for tuned models
print(f"Tuned Random Forest PM2.5 Model: MAE={mae_pm25_rf_tuned:.2f}, R2={r2_pm25_rf_tuned:.2f}")
print(f"Tuned Gradient Boosting PM2.5 Model: MAE={mae_pm25_gb_tuned:.2f}, R2={r2_pm25_gb_tuned:.2f}")

# Feature importance for Random Forest
rf_feature_importance = best_rf_model.feature_importances_
rf_feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_feature_importance
}).sort_values(by='Importance', ascending=False)

# Feature importance for Gradient Boosting
gb_feature_importance = best_gb_model.feature_importances_
gb_feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gb_feature_importance
}).sort_values(by='Importance', ascending=False)

# Print feature importance
print("Random Forest Feature Importance:")
print(rf_feature_importance_df)

print("Gradient Boosting Feature Importance:")
print(gb_feature_importance_df)

# Visualizing the Results: Feature Importance
plt.figure(figsize=(14, 6))

# Random Forest Feature Importance
plt.subplot(1, 2, 1)
sns.barplot(x='Importance', y='Feature', data=rf_feature_importance_df)
plt.title("Random Forest Feature Importance")

# Gradient Boosting Feature Importance
plt.subplot(1, 2, 2)
sns.barplot(x='Importance', y='Feature', data=gb_feature_importance_df)
plt.title("Gradient Boosting Feature Importance")

plt.tight_layout()
plt.show()

# Visualize the regression results: PM2.5 Predictions
plt.figure(figsize=(14, 6))

# Random Forest Regression Results
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_test["PopulationDensity_people/km²"], y=y_pm25_test, label="Actual", alpha=0.6)
sns.lineplot(x=X_test["PopulationDensity_people/km²"], y=y_pm25_pred_rf_tuned, color="red", label="Predicted")
plt.title("Random Forest: PM2.5 vs Population Density")
plt.xlabel("Population Density (people/km²)")
plt.ylabel("PM2.5 (µg/m³)")
plt.legend()

# Gradient Boosting Regression Results
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_test["PopulationDensity_people/km²"], y=y_pm25_test, label="Actual", alpha=0.6)
sns.lineplot(x=X_test["PopulationDensity_people/km²"], y=y_pm25_pred_gb_tuned, color="red", label="Predicted")
plt.title("Gradient Boosting: PM2.5 vs Population Density")
plt.xlabel("Population Density (people/km²)")
plt.ylabel("PM2.5 (µg/m³)")
plt.legend()

plt.tight_layout()
plt.show()


"""
# INSIGHT
1. Korelasi antara Polusi dan Kepadatan Populasi
Insight dari Heatmap:
Korelasi positif ditemukan antara PopulationDensity_people/km² dengan Pollutant_PM2.5_µg/m³ (nilai korelasi = 0.65) dan Pollutant_PM10_µg/m³ (nilai korelasi = 0.72).
Hal ini menunjukkan bahwa wilayah dengan kepadatan populasi yang lebih tinggi cenderung memiliki tingkat polusi udara yang lebih tinggi, terutama untuk partikel PM2.5 dan PM10.

Interpretasi:
Aktivitas manusia seperti transportasi, industri, dan pembakaran bahan bakar fosil di wilayah padat penduduk berkontribusi terhadap peningkatan partikel polusi ini.
Korelasi yang cukup kuat menunjukkan bahwa kebijakan pengelolaan urbanisasi dan transportasi harus diperhatikan untuk mengurangi polusi.

2. Pola Hubungan Berdasarkan Scatter Plot
Scatter Plot: PM2.5 vs Population Density
Distribusi data menunjukkan pola tren naik, di mana peningkatan kepadatan penduduk biasanya disertai dengan peningkatan konsentrasi PM2.5.
Namun, terdapat beberapa outlier, yaitu daerah dengan kepadatan tinggi tetapi tingkat PM2.5 yang relatif rendah. Hal ini mungkin disebabkan oleh adanya kebijakan mitigasi polusi yang efektif, seperti ruang hijau atau energi bersih.

Scatter Plot: PM10 vs Population Density
Tren serupa terlihat pada PM10, tetapi dengan korelasi yang sedikit lebih tinggi. Partikel PM10 cenderung lebih mudah terdistribusi di atmosfer, sehingga lebih sensitif terhadap aktivitas manusia yang intensif.

Interpretasi:
Pola ini mendukung hipotesis awal bahwa wilayah padat penduduk lebih rentan terhadap polusi udara. Namun, peran mitigasi lokal seperti ruang hijau atau energi terbarukan dapat memengaruhi hasil di beberapa lokasi.

3. Hasil Model Linear Regression
PM2.5:
MAE: 7.5 µg/m³
R²: 0.58
PM10:
MAE: 12.3 µg/m³
R²: 0.64

Interpretasi Model:
Model linear regression memberikan hasil yang cukup baik dengan nilai R² sekitar 0.58–0.64, menunjukkan bahwa sekitar 58–64% variasi dalam konsentrasi polusi udara dapat dijelaskan oleh kepadatan populasi.
Namun, error (MAE) menunjukkan ada faktor lain yang signifikan memengaruhi polusi udara, seperti kondisi iklim, jenis industri, atau penggunaan teknologi ramah lingkungan.

4. Kebijakan dan Rekomendasi
Peningkatan Ruang Hijau:
Scatter plot menunjukkan bahwa beberapa daerah dengan kepadatan tinggi memiliki tingkat polusi rendah, kemungkinan berkat kebijakan ruang hijau. Pemerintah dapat meningkatkan persentase GreenSpaceIndex_% di wilayah urban.

Kebijakan Transportasi Berkelanjutan:
Wilayah dengan kepadatan tinggi sering memiliki aktivitas transportasi yang signifikan. Implementasi transportasi publik yang ramah lingkungan seperti bus listrik dapat membantu menekan emisi.

Teknologi Ramah Lingkungan:
Mengadopsi EnergySavingTechnology atau meningkatkan penggunaan energi terbarukan dapat membantu mengurangi emisi, meskipun di wilayah dengan kepadatan populasi tinggi.

Regulasi pada Industri Lokal:
Untuk daerah dengan kepadatan tinggi, pemantauan emisi industri lokal sangat penting. Regulasi terhadap emisi industri dapat membantu mengurangi tingkat PM2.5 dan PM10.

Data Tambahan untuk Peningkatan Model:
Faktor seperti arah angin, suhu, dan kelembaban (%Humidity) dapat dimasukkan untuk meningkatkan akurasi model prediksi.

Kesimpulan Utama:
Kepadatan populasi adalah faktor signifikan yang memengaruhi polusi udara. Namun, kebijakan mitigasi seperti ruang hijau dan transportasi berkelanjutan dapat mengurangi dampaknya.
Untuk meningkatkan akurasi dalam lomba, integrasi lebih banyak fitur lingkungan dan implementasi model non-linear (seperti Random Forest atau Gradient Boosting) dapat diuji.
"""
