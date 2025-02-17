import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
environmental_df = pd.read_csv('NEW_ENVIRONMENT_DATASET.csv')
building_df = pd.read_csv('NEW_BUILDING_DATASET.csv')

# Gabungkan data berdasarkan suhu udara (contoh kolom penggabungan, sesuaikan jika perlu)
merged_df = pd.merge(
    environmental_df,
    building_df,
    left_on='AirTemperature_C',
    right_on='WeatherData_Temperature_C',
    how='inner'
)

# Analisis 1: Hubungan Polusi dengan Kepadatan Populasi
plt.scatter(merged_df['PopulationDensity_people/km²'], merged_df['Pollutant_PM2.5_µg/m³'], alpha=0.6, label='PM2.5')
plt.scatter(merged_df['PopulationDensity_people/km²'], merged_df['Pollutant_PM10_µg/m³'], alpha=0.6, label='PM10', color='orange')
plt.title('Hubungan Kepadatan Populasi dengan Polusi Udara')
plt.xlabel('Kepadatan Populasi (people/km²)')
plt.ylabel('Konsentrasi Polutan (µg/m³)')
plt.legend()
plt.show()

# Analisis 2: Hubungan Konsumsi Energi dengan Kapasitas Energi Terbarukan
plt.scatter(merged_df['RenewableCapacity_kWh'], merged_df['MonthlyElectricityConsumption_kWh'], alpha=0.6)
plt.title('Hubungan Kapasitas Energi Terbarukan dengan Konsumsi Energi')
plt.xlabel('Kapasitas Energi Terbarukan (kWh)')
plt.ylabel('Konsumsi Listrik Bulanan (kWh)')
plt.show()

# Analisis 3: Hubungan Efisiensi Energi dengan Teknologi Hemat Energi
grouped_tech = merged_df.groupby('EnergySavingTechnology')['EnergyEfficiency_kWh_per_m2'].mean()
grouped_tech.plot(kind='bar', color='teal')
plt.title('Efisiensi Energi berdasarkan Teknologi Hemat Energi')
plt.xlabel('Teknologi Hemat Energi')
plt.ylabel('Efisiensi Energi (kWh/m²)')
plt.xticks(rotation=45)
plt.show()

# Analisis 4: Hubungan Ruang Hijau dengan Kualitas Udara
plt.scatter(merged_df['GreenSpaceIndex_%'], merged_df['Pollutant_PM2.5_µg/m³'], alpha=0.6, label='PM2.5')
plt.scatter(merged_df['GreenSpaceIndex_%'], merged_df['Pollutant_PM10_µg/m³'], alpha=0.6, label='PM10', color='orange')
plt.title('Hubungan Ruang Hijau dengan Polusi Udara')
plt.xlabel('Ruang Hijau (%)')
plt.ylabel('Konsentrasi Polutan (µg/m³)')
plt.legend()
plt.show()

# Analisis hubungan AQI dengan berbagai polutan
pollutants = ['Pollutant_PM2.5_µg/m³', 'Pollutant_PM10_µg/m³', 'Pollutant_O3_ppb',
              'Pollutant_NO2_ppb', 'Pollutant_CO_ppm', 'Pollutant_SO2_ppb']

# Visualisasi hubungan AQI dengan masing-masing polutan
for pollutant in pollutants:
    plt.scatter(merged_df[pollutant], merged_df['AQI_Index'], label=pollutant, alpha=0.6)
    plt.title(f'Hubungan {pollutant} dengan AQI')
    plt.xlabel(pollutant)
    plt.ylabel('AQI Index')
    plt.show()

# Korelasi tambahan untuk konfirmasi hubungan
populasi_polusi_corr = merged_df[['PopulationDensity_people/km²', 'Pollutant_PM2.5_µg/m³', 'Pollutant_PM10_µg/m³']].corr()
energi_terbarukan_konsumsi_corr = merged_df[['RenewableCapacity_kWh', 'MonthlyElectricityConsumption_kWh']].corr()
ruang_hijau_polusi_corr = merged_df[['GreenSpaceIndex_%', 'Pollutant_PM2.5_µg/m³', 'Pollutant_PM10_µg/m³']].corr()
aqi_pollution_corr = merged_df[['AQI_Index'] + pollutants].corr()

print("Korelasi AQI dengan berbagai polutan:")
print(aqi_pollution_corr['AQI_Index'])
print("Korelasi Kepadatan Populasi dan Polusi:")
print(populasi_polusi_corr)
print("\nKorelasi Kapasitas Energi Terbarukan dan Konsumsi Energi:")
print(energi_terbarukan_konsumsi_corr)
print("\nKorelasi Ruang Hijau dan Polusi:")
print(ruang_hijau_polusi_corr)

"""
1. Hubungan Kepadatan Populasi dengan Polusi Udara
Grafik: Terdapat dua scatter plot yang menunjukkan hubungan antara kepadatan populasi (people/km²) dengan konsentrasi polutan PM2.5 dan PM10.
Penjelasan:
Grafik ini menunjukkan bagaimana kepadatan penduduk berhubungan dengan tingkat polusi udara.
Dari grafik, terlihat bahwa semakin tinggi kepadatan populasi, semakin tinggi konsentrasi polutan seperti PM2.5 dan PM10. Hal ini mungkin disebabkan oleh faktor-faktor seperti peningkatan kendaraan bermotor, lebih banyak aktivitas industri, dan sumber polusi lainnya di area dengan kepadatan penduduk yang tinggi.

2. Hubungan Kapasitas Energi Terbarukan dengan Konsumsi Energi
Grafik: Scatter plot yang menunjukkan hubungan antara Kapasitas Energi Terbarukan (kWh) dengan Konsumsi Energi Bulanan (kWh).
Penjelasan:
Grafik ini menunjukkan bahwa tidak ada hubungan yang kuat antara kapasitas energi terbarukan dan konsumsi energi bulanan. Korelasi yang sangat rendah (0.003094) menunjukkan bahwa meskipun kapasitas energi terbarukan ada, hal itu tidak secara langsung mempengaruhi konsumsi energi yang tercatat.
Kemungkinan, ini disebabkan oleh ketergantungan pada sumber energi lainnya yang masih mendominasi konsumsi energi.

3. Efisiensi Energi berdasarkan Teknologi Hemat Energi
Grafik: Bar chart yang menunjukkan efisiensi energi berdasarkan teknologi hemat energi yang diterapkan.
Penjelasan:
Teknologi seperti LED Lighting dan Efficient HVAC menunjukkan efisiensi energi yang lebih tinggi dibandingkan dengan teknologi lainnya seperti Solar Panels dan Smart Thermostats.
Ini memberikan wawasan bahwa penerapan teknologi hemat energi dapat memiliki dampak signifikan terhadap efisiensi energi bangunan, dan beberapa teknologi mungkin lebih efisien daripada yang lain tergantung pada kondisi lingkungan dan bangunan.

4. Hubungan Ruang Hijau dengan Polusi Udara
Grafik: Scatter plot yang menunjukkan hubungan antara GreenSpaceIndex (%) dan polutan PM2.5 serta PM10.
Penjelasan:
Ruang hijau di suatu area berfungsi sebagai penyerapan polusi udara. Meskipun korelasi yang terlihat di grafik ini sangat lemah, menunjukkan bahwa adanya ruang hijau dapat sedikit mengurangi konsentrasi polusi udara seperti PM2.5 dan PM10, tetapi dampaknya tidak selalu signifikan di semua kasus.
Hal ini bisa disebabkan oleh kurangnya ruang hijau yang cukup atau polusi dari sumber lain yang lebih dominan.

5. Hubungan AQI dengan Berbagai Polutan
Grafik: Scatter plot yang menunjukkan hubungan antara AQI (Air Quality Index) dan berbagai jenis polutan (PM2.5, PM10, O3, NO2, CO, SO2).
Penjelasan:
AQI berfungsi sebagai indikator umum kualitas udara, dan grafik menunjukkan hubungan yang lebih kuat antara AQI dan polutan PM2.5 (korelasi 0.78), yang merupakan salah satu polutan utama yang memengaruhi kualitas udara.
Polutan lainnya, seperti PM10 dan NO2, juga menunjukkan korelasi moderat dengan AQI, menunjukkan bahwa semakin tinggi konsentrasi polutan ini, semakin buruk kualitas udara yang tercermin dalam AQI.

6. Korelasi Tambahan
Korelasi AQI dengan Polutan:
PM2.5 memiliki korelasi tertinggi dengan AQI, yang menunjukkan bahwa PM2.5 adalah polutan yang paling berpengaruh terhadap indeks kualitas udara.
Korelasi Kepadatan Populasi dengan Polusi:
Kepadatan populasi memiliki korelasi positif dengan PM2.5 dan PM10, menunjukkan bahwa area dengan kepadatan tinggi cenderung memiliki polusi yang lebih tinggi.
Korelasi Kapasitas Energi Terbarukan dengan Konsumsi Energi:
Korelasi yang sangat rendah menunjukkan bahwa kapasitas energi terbarukan tidak berhubungan langsung dengan konsumsi energi dalam dataset ini.
Korelasi Ruang Hijau dengan Polusi:
Tidak ada korelasi kuat antara ruang hijau dan polusi udara, meskipun diharapkan ruang hijau dapat membantu mengurangi polusi udara.

Insight Umum:
- Kepadatan populasi berhubungan positif dengan polusi udara, sehingga area dengan kepadatan tinggi cenderung memiliki polusi yang lebih buruk.
- Kapasitas energi terbarukan saat ini belum menunjukkan dampak besar terhadap pengurangan konsumsi energi, menunjukkan pentingnya peningkatan implementasi dan efisiensi sistem energi terbarukan.
- Ruang hijau menunjukkan pengaruh yang relatif kecil terhadap penurunan polusi udara dalam dataset ini, meskipun biasanya ruang hijau dianggap dapat menyerap polutan.
- AQI sangat dipengaruhi oleh PM2.5, yang mengindikasikan bahwa PM2.5 merupakan polutan utama yang harus diatasi untuk memperbaiki kualitas udara.

Langkah Selanjutnya:
1. Untuk meningkatkan kualitas udara di daerah dengan kepadatan populasi tinggi, strategi mitigasi polusi seperti penggunaan transportasi umum atau kendaraan ramah lingkungan serta peningkatan kapasitas ruang hijau bisa dipertimbangkan.
2. Implementasi teknologi hemat energi juga harus diprioritaskan untuk mengurangi konsumsi energi dan polusi terkait.
"""