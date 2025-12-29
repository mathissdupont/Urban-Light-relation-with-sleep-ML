"""src/visualization/dashboard_app.py

Bu dosya, Streamlit + Folium kullanarak etkileşimli bir risk haritası dashboard'u oluşturur.

Amaç
- `final_model_dataset.csv` içindeki feature/label'ları, `grid_cells.geojson` geometrileriyle birleştirmek.
- Kullanıcıya filtreleme imkanı vermek:
    - sadece yüksek risk hücreleri
    - ışık yoğunluğu (night_light_avg) aralığı
- Harita üzerinde her grid hücresini centroid noktasında nokta olarak göstermek:
    - kırmızı: high_noise_risk = 1
    - yeşil: high_noise_risk = 0

Girdiler
- data/processed/final_model_dataset.csv
- data/processed/grid_cells.geojson

Çalıştırma
- Proje kökünde: `streamlit run src/visualization/dashboard_app.py`

Not
- Folium harita lat/lon beklediği için grid WGS84'e (EPSG:4326) dönüştürülür.
- Merge sonrası bazı hücrelerde NaN olabilir (ör. veri eksikliği); popup alanları güvenli şekilde doldurulur.
"""

import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_model_dataset.csv"
GRID_PATH = PROJECT_ROOT / "data" / "processed" / "grid_cells.geojson"

st.set_page_config(layout="wide")
st.title("NYC Nighttime Noise Risk Dashboard (Light + 311 + ML)")

df = pd.read_csv(DATA_PATH)
grid = gpd.read_file(GRID_PATH)

# Grid'i WGS84'e al (Folium haritası lat/lon ister)
if grid.crs is None:
    st.error("Grid CRS bulunamadı. grid_cells.geojson CRS içermiyor olabilir.")
    st.stop()

grid_wgs = grid.to_crs("EPSG:4326")

# Centroid'i geometry'den hesapla (grid dosyasında centroid kolonları olsa bile,
# burada bağımsız ve garanti bir yöntem olarak geometry'den üretip kullanıyoruz)
centroids = grid_wgs.geometry.centroid
grid_wgs["lat"] = centroids.y
grid_wgs["lon"] = centroids.x

# Merge (cell_id ortak anahtar): geometri + model tablosu aynı satırda birleşir
gdf = grid_wgs.merge(df, on="cell_id", how="left")

# Sidebar filtreleri: haritayı daraltarak okunabilirliği artırır
st.sidebar.header("Filtreler")
show_only_high = st.sidebar.checkbox("Sadece yüksek risk (1) göster", value=False)

# Opsiyonel: ışık yoğunluğuna göre filtre (min/max değerleri dataset'ten gelir)
min_light, max_light = float(gdf["night_light_avg"].min()), float(gdf["night_light_avg"].max())
light_range = st.sidebar.slider("Night light avg aralığı", min_light, max_light, (min_light, max_light))

# Filtre uygula
gdf = gdf[(gdf["night_light_avg"] >= light_range[0]) & (gdf["night_light_avg"] <= light_range[1])]
if show_only_high:
    gdf = gdf[gdf["high_noise_risk"] == 1]

# Harita tabanı
m = folium.Map(location=[40.73, -73.94], zoom_start=11, tiles="cartodbpositron")

# Noktaları çiz: centroid noktasında küçük marker'lar
for _, row in gdf.iterrows():
    # bazı hücrelerde df merge sonrası NaN olabilir, güvenli geçelim
    risk = int(row["high_noise_risk"]) if pd.notna(row["high_noise_risk"]) else 0
    color = "red" if risk == 1 else "green"

    noise_cnt = int(row["noise_night_count"]) if pd.notna(row["noise_night_count"]) else 0
    light_val = float(row["night_light_avg"]) if pd.notna(row["night_light_avg"]) else 0.0

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=3,
        color=color,
        fill=True,
        fill_opacity=0.65,
        popup=folium.Popup(
            f"""
            <b>Cell:</b> {int(row['cell_id'])}<br>
            <b>Risk:</b> {risk}<br>
            <b>Night complaints:</b> {noise_cnt}<br>
            <b>Night light avg:</b> {light_val:.2f}
            """,
            max_width=250
        )
    ).add_to(m)

st_folium(m, width=1200, height=700)

st.caption("Kırmızı: yüksek gece gürültü riski (top %20 complaint yoğunluğu), Yeşil: düşük risk.")
