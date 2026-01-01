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


@st.cache_data(show_spinner=False)
def _load_inputs(data_path: Path, grid_path: Path) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    df_local = pd.read_csv(data_path)
    grid_local = gpd.read_file(grid_path)
    return df_local, grid_local


def _coerce_cell_id(series: pd.Series) -> pd.Series:
    # cell_id farklı dosyalarda int/str gelebiliyor; merge'in tutarlı olması için numeriğe çeviriyoruz.
    # Int64: NA destekli integer.
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _safe_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


df, grid = _load_inputs(DATA_PATH, GRID_PATH)

if not DATA_PATH.exists():
    st.error(f"Dataset bulunamadı: {DATA_PATH}")
    st.stop()
if not GRID_PATH.exists():
    st.error(f"Grid bulunamadı: {GRID_PATH}")
    st.stop()

# Grid CRS kontrolü (Folium lat/lon ister)
if grid.crs is None:
    st.error("Grid CRS bulunamadı. grid_cells.geojson CRS içermiyor olabilir.")
    st.stop()

# cell_id tiplerini normalize et
if "cell_id" not in df.columns:
    st.error("final_model_dataset.csv içinde 'cell_id' kolonu yok.")
    st.stop()
if "cell_id" not in grid.columns:
    st.error("grid_cells.geojson içinde 'cell_id' kolonu yok.")
    st.stop()

df = df.copy()
grid = grid.copy()
df["cell_id"] = _coerce_cell_id(df["cell_id"])
grid["cell_id"] = _coerce_cell_id(grid["cell_id"])

# Folium için WGS84
grid_wgs = grid.to_crs("EPSG:4326")

# Nokta koordinatları: mümkünse hazır centroid_lat/lon'u kullan, yoksa güvenli centroid hesapla.
if {"centroid_lat", "centroid_lon"}.issubset(set(grid.columns)):
    grid_wgs["lat"] = _safe_float_series(grid_wgs["centroid_lat"])
    grid_wgs["lon"] = _safe_float_series(grid_wgs["centroid_lon"])
else:
    # Geographic CRS üzerinde centroid uyarısını/yanlışlığını azaltmak için önce metrik CRS'e al.
    grid_metric = grid.to_crs("EPSG:3857")
    centroids_metric = grid_metric.geometry.centroid
    centroids_wgs = gpd.GeoSeries(centroids_metric, crs="EPSG:3857").to_crs("EPSG:4326")
    grid_wgs["lat"] = centroids_wgs.y
    grid_wgs["lon"] = centroids_wgs.x

gdf = grid_wgs.merge(df, on="cell_id", how="left")

FEATURES_USED_IN_BASELINE = [
    "night_light_avg",
    "centroid_lat",
    "centroid_lon",
    "peak_hour",
]

required_cols = {"night_light_avg", "high_noise_risk"}
missing = sorted(required_cols - set(gdf.columns))
if missing:
    st.error(f"Dashboard için gereken kolon(lar) eksik: {', '.join(missing)}")
    st.stop()

# Tip güvenliği
gdf["night_light_avg"] = _safe_float_series(gdf["night_light_avg"])
gdf["high_noise_risk"] = pd.to_numeric(gdf["high_noise_risk"], errors="coerce").fillna(0).astype(int)
if "noise_night_count" in gdf.columns:
    gdf["noise_night_count"] = pd.to_numeric(gdf["noise_night_count"], errors="coerce").fillna(0).astype(int)
if "peak_hour" in gdf.columns:
    gdf["peak_hour"] = pd.to_numeric(gdf["peak_hour"], errors="coerce")


with st.expander("Proje özeti (bu proje ne yapıyor?)", expanded=True):
    st.markdown(
        """
Bu proje, NYC genelinde **gece gürültü riski**ni mekansal olarak haritalamak için 2 ana kaynağı aynı grid üzerinde birleştirir:

- **311 (Noise - Residential)** şikayetleri (22:00–06:00 gece filtresi)
- **VIIRS Night Lights** gece ışık şiddeti

Akış (özet):
- 500m'lik grid üretilir → `data/processed/grid_cells.geojson`
- 311 noktaları grid'e sayılır → `noise_night_count`
- VIIRS rasterı grid'e özetlenir → `night_light_avg`
- Son tabloda label üretilir:
  - `high_noise_risk = 1` ⇢ `noise_night_count` üst %20 dilimdeyse (80. persentil)

Bu dashboard, her grid hücresini centroid noktasında gösterir:
- **Kırmızı:** yüksek risk (1)
- **Yeşil:** düşük risk (0)
        """
    )

with st.expander("Feature selection / kullanılan feature'lar", expanded=False):
    st.markdown(
        """
Bu projede şu an **otomatik feature selection** (RFE / SelectKBest / L1-eleme vb.) yok.

Baseline model eğitiminde kullanılan feature seti **manuel** olarak tanımlı:
- `night_light_avg` (VIIRS gece ışık ortalaması)
- `centroid_lat`, `centroid_lon` (hücrenin konumu)
- `peak_hour` (gece şikayetlerinin en yoğun olduğu saat; yoksa -1)

Bu seti model tarafında şu dosyalar kullanıyor:
- `src/models/train_baseline.py`
- `src/models/plot_confusion_matrices.py`
- `src/analysis/plot_feature_importance_rf.py`
        """
    )

    st.write("Baseline feature listesi:")
    st.dataframe(pd.DataFrame({"feature": FEATURES_USED_IN_BASELINE}))

    fi_path = PROJECT_ROOT / "outputs" / "figures" / "rf_feature_importance.png"
    if fi_path.exists():
        st.write("Random Forest feature importance (varsa):")
        st.image(str(fi_path), use_container_width=True)
    else:
        st.info(
            "Feature importance görseli henüz üretilmemiş. Üretmek için: "
            "`python src/analysis/plot_feature_importance_rf.py`"
        )

# Basit KPI'lar
total_cells = int(gdf["cell_id"].notna().sum())
high_cells = int((gdf["high_noise_risk"] == 1).sum())
rate = (high_cells / total_cells * 100.0) if total_cells else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("Toplam grid hücresi", f"{total_cells:,}")
col2.metric("Yüksek risk hücresi", f"{high_cells:,}")
col3.metric("Yüksek risk oranı", f"{rate:.1f}%")

# Sidebar filtreleri: haritayı daraltarak okunabilirliği artırır
st.sidebar.header("Filtreler")
show_only_high = st.sidebar.checkbox("Sadece yüksek risk (1) göster", value=False)

# Opsiyonel: ışık yoğunluğuna göre filtre (NaN-safe)
light_valid = gdf["night_light_avg"].dropna()
if light_valid.empty:
    st.sidebar.warning("night_light_avg değerleri bulunamadı; ışık filtresi devre dışı.")
    light_range = (0.0, 0.0)
else:
    min_light, max_light = float(light_valid.min()), float(light_valid.max())
    light_range = st.sidebar.slider(
        "Night light avg aralığı",
        min_value=min_light,
        max_value=max_light,
        value=(min_light, max_light),
    )

# Filtre uygula
if not light_valid.empty:
    gdf = gdf[(gdf["night_light_avg"] >= light_range[0]) & (gdf["night_light_avg"] <= light_range[1])]
if show_only_high:
    gdf = gdf[gdf["high_noise_risk"] == 1]

# Harita tabanı
m = folium.Map(location=[40.73, -73.94], zoom_start=11, tiles="cartodbpositron")

if gdf.empty:
    st.warning("Seçilen filtrelerle gösterilecek hücre kalmadı. Filtreleri gevşetmeyi deneyin.")
else:
    # Noktaları çiz: centroid noktasında küçük marker'lar
    for _, row in gdf.iterrows():
        risk = int(row["high_noise_risk"]) if pd.notna(row["high_noise_risk"]) else 0
        color = "red" if risk == 1 else "green"

        noise_cnt = int(row["noise_night_count"]) if ("noise_night_count" in row and pd.notna(row["noise_night_count"])) else 0
        light_val = float(row["night_light_avg"]) if pd.notna(row["night_light_avg"]) else 0.0

        lat = float(row["lat"]) if pd.notna(row["lat"]) else None
        lon = float(row["lon"]) if pd.notna(row["lon"]) else None
        if lat is None or lon is None:
            continue

        cell_display = row["cell_id"]
        try:
            cell_display = int(cell_display)
        except Exception:
            cell_display = str(cell_display)

        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.65,
            popup=folium.Popup(
                f"""
                <b>Cell:</b> {cell_display}<br>
                <b>Risk:</b> {risk}<br>
                <b>Night complaints:</b> {noise_cnt}<br>
                <b>Night light avg:</b> {light_val:.2f}
                """,
                max_width=250
            )
        ).add_to(m)

st_folium(m, width=1200, height=700)

st.caption("Kırmızı: yüksek gece gürültü riski (top %20 complaint yoğunluğu), Yeşil: düşük risk.")
