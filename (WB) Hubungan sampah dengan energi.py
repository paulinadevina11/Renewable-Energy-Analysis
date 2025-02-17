import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load datasets
waste_data = pd.read_csv("(WB) Sampah tiap negara.csv")
renewable_data = pd.read_csv("(WB) Renewable Energy tiap negara.csv")

# Rename columns for consistency
waste_data = waste_data.rename(columns={"country_name": "Country"})
renewable_data = renewable_data.rename(columns={"Entity": "Country"})

# Drop rows with missing values in relevant columns
waste_data = waste_data.dropna(subset=["total_msw_total_msw_generated_tons_year"])
renewable_data = renewable_data.dropna(subset=["Renewables (% equivalent primary energy)"])

# Filter renewable data for years >= 2014
renewable_filtered = renewable_data[renewable_data["Year"] >= 2014]

# Select the latest year for each country in the filtered renewable data
renewable_latest = renewable_filtered.loc[renewable_filtered.groupby("Country")["Year"].idxmax()]

# Select relevant columns
waste_data_cleaned = waste_data[["Country", "total_msw_total_msw_generated_tons_year"]]
renewable_data_cleaned = renewable_latest[["Country", "Year", "Renewables (% equivalent primary energy)"]]

# Merge datasets on Country
merged_data = pd.merge(waste_data_cleaned, renewable_data_cleaned, on="Country", how="inner")

merged_data = merged_data.rename(columns={
    "total_msw_total_msw_generated_tons_year": "Total_MSW_Tons_Per_Year",
    "Renewables (% equivalent primary energy)": "Renewable_Energy_Percentage"
})

# Print the merged data
print("Merged Data (Filtered and Cleaned):")
print(merged_data.head())

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(merged_data[["Total_MSW_Tons_Per_Year", "Renewable_Energy_Percentage"]])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
merged_data["Cluster"] = kmeans.fit_predict(scaled_data)

# Print the cluster data
print("\nData with Cluster Assignments:")
print(merged_data)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(merged_data[["Total_MSW_Tons_Per_Year", "Renewable_Energy_Percentage"]])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
merged_data["Cluster"] = kmeans.fit_predict(scaled_data)

# Print the cluster data
print("\nData with Cluster Assignments:")
print(merged_data)

# Group data by cluster and print summary
cluster_summary = merged_data.groupby("Cluster")[["Total_MSW_Tons_Per_Year", "Renewable_Energy_Percentage"]].mean()
print("\nCluster Summary (Mean Values):")
print(cluster_summary)

# Print countries in each cluster
print("\nCountries in Each Cluster:")
for cluster in merged_data["Cluster"].unique():
    countries_in_cluster = merged_data[merged_data["Cluster"] == cluster]["Country"].tolist()
    print(f"Cluster {cluster}: {countries_in_cluster}")

# Plot the clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=merged_data,
    x="Total_MSW_Tons_Per_Year",
    y="Renewable_Energy_Percentage",
    hue="Cluster",
    palette="Set2",
    size="Total_MSW_Tons_Per_Year",
    sizes=(20, 200),
    alpha=0.7
)
plt.title("Cluster Analysis: Renewable Energy vs Total MSW (Filtered Data)", fontsize=16)
plt.xlabel("Total MSW (Tons Per Year)", fontsize=12)
plt.ylabel("Renewable Energy Percentage", fontsize=12)
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()

# Visualize Renewable Energy Percentage (1st Dimension)
plt.figure(figsize=(10, 6))
for cluster in merged_data['Cluster'].unique():
    cluster_data = merged_data[merged_data['Cluster'] == cluster]
    plt.scatter(cluster_data.index, cluster_data['Renewable_Energy_Percentage'], label=f'Cluster {cluster}')
plt.title('Cluster Analysis: Renewable Energy Percentage by Country')
plt.xlabel('Countries (Index)')
plt.ylabel('Renewable Energy Percentage')
plt.legend()
plt.show()

# Visualize Total MSW (2nd Dimension)
plt.figure(figsize=(10, 6))
for cluster in merged_data['Cluster'].unique():
    cluster_data = merged_data[merged_data['Cluster'] == cluster]
    plt.scatter(cluster_data.index, cluster_data['Total_MSW_Tons_Per_Year'], label=f'Cluster {cluster}')
plt.title('Cluster Analysis: Total MSW by Country')
plt.xlabel('Countries (Index)')
plt.ylabel('Total MSW (Tons Per Year)')
plt.legend()
plt.show()

# Visualize Combined Clusters (MSW vs Renewable Energy)
plt.figure(figsize=(10, 6))
for cluster in merged_data['Cluster'].unique():
    cluster_data = merged_data[merged_data['Cluster'] == cluster]
    plt.scatter(cluster_data['Total_MSW_Tons_Per_Year'], cluster_data['Renewable_Energy_Percentage'], label=f'Cluster {cluster}')
plt.title('Cluster Analysis: Total MSW vs Renewable Energy')
plt.xlabel('Total MSW (Tons Per Year)')
plt.ylabel('Renewable Energy Percentage')
plt.legend()
plt.show()


"""
Inti Permasalahan

Banyak negara menghasilkan limbah perkotaan (Municipal Solid Waste/MSW) dalam jumlah besar
tetapi belum mengintegrasikan solusi energi terbarukan yang efektif. Hal ini menciptakan tantangan besar bagi
keberlanjutan lingkungan, terutama di negara-negara yang limbahnya sangat tinggi tetapi kontribusi energi terbarukan
masih rendah.

Ketimpangan antar negara dalam pengelolaan limbah dan penggunaan energi terbarukan:
Ada disparitas yang jelas antara negara-negara maju dan berkembang, di mana negara maju cenderung memiliki
infrastruktur yang lebih baik untuk memanfaatkan energi terbarukan, sementara negara berkembang seringkali terjebak
dengan sistem energi berbasis bahan bakar fosil.

Insight Utama
Cluster 1 (China, AS, India):
Negara-negara ini menghasilkan limbah dalam jumlah besar tetapi memiliki kontribusi energi terbarukan yang rendah (~11.6%).
Insight:
Mereka memiliki peluang besar untuk mengembangkan inisiatif waste-to-energy, di mana limbah padat dapat diubah menjadi sumber energi terbarukan. Hal ini dapat membantu menurunkan tingkat limbah sekaligus meningkatkan porsi energi terbarukan.

Cluster 0 (Austria, Swedia, Norwegia, dll):
Negara-negara ini memiliki kontribusi energi terbarukan tinggi (~40.9%) dengan tingkat limbah relatif rendah.
Insight:
Kebijakan keberlanjutan mereka efektif, dan mereka dapat menjadi model bagi negara lain dalam hal kebijakan energi dan pengelolaan limbah.

Cluster 2 (Negara berkembang):
Negara-negara ini memiliki limbah yang sedang hingga rendah tetapi kontribusi energi terbarukan masih sangat kecil (~9.3%).
Insight:
Infrastruktur energi terbarukan masih lemah. Integrasi teknologi pengelolaan limbah untuk energi dapat menjadi solusi awal yang feasibel.

Rekomendasi Solusi

Peningkatan Infrastruktur Waste-to-Energy:
Negara-negara dengan produksi limbah tinggi (Cluster 1) harus fokus pada investasi dalam teknologi yang mengubah limbah menjadi energi, mengurangi ketergantungan pada energi fosil.

Transfer Pengetahuan dari Cluster 0 ke Cluster 2:
Negara-negara dengan kontribusi energi terbarukan tinggi (Cluster 0) dapat berbagi teknologi dan kebijakan untuk membantu negara berkembang meningkatkan efisiensi energi mereka.

Prioritaskan Kebijakan Energi Terbarukan untuk Negara Berkembang:
Dukungan internasional dalam bentuk pendanaan dan teknologi dapat mempercepat transisi negara Cluster 2 ke model energi terbarukan yang lebih berkelanjutan.

Pengelolaan Limbah yang Efisien:
Optimalisasi pengumpulan, daur ulang, dan pemanfaatan limbah akan membantu negara dalam semua cluster mengurangi dampak lingkungan mereka.
"""
