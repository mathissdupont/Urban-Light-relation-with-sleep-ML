# Urban Light + Sleep ML — Kod Açıklamaları (TR)

Bu doküman, projede “nerede ne yaptık?” sorusunu uçtan uca cevaplamak için hazırlanmıştır.

## 1) Genel Mimari

Bu proje, NYC ölçeğinde 2 ana veri kaynağını ortak bir mekansal grid üzerinde birleştirip, **gece gürültü riski** için basit bir ML baseline kurar:

- **311 (Noise - Residential)** şikayetleri (noktasal olay verisi)
- **VIIRS Night Lights** (raster) gece ışık şiddeti

Ortak anahtar: `cell_id` (grid hücre id’si).

Üretilen temel çıktılar:
- Grid: `data/processed/grid_cells.geojson`
- Grid bazında 311 sayımları: `data/processed/grid_311_counts.csv`
- Grid bazında VIIRS ışık özeti: `data/processed/grid_viirs_light.csv`
- Nihai model dataseti: `data/processed/final_model_dataset.csv`

## 2) Veri Akışı (Pipeline)

### A) Grid üretimi

Dosya: [src/data/create_grid.py](../src/data/create_grid.py)

Ne yapar?
- NYC borough sınırlarını okur.
- Metre CRS’e dönüştürür (`EPSG:32618`).
- Bounding box üzerinde 500m’lik kare fishnet üretir.
- NYC sınırına göre clip’ler.
- Her hücreye `cell_id` verir ve centroid lat/lon üretir.

Çıktı:
- `data/processed/grid_cells.geojson`

### B) 311 verisini indirme ve gece filtresi

Dosya: [download_311_filtered.py](../download_311_filtered.py)

Ne yapar?
- Socrata üzerinden 311 verisini parça parça indirir (LIMIT/OFFSET).
- Sadece `Noise - Residential`, 2020 sonrası kayıtlar.
- Gece filtresi uygular: 22:00–06:00.

Çıktı:
- `nyc_311_noise_night_2020_present.csv` (scriptin çalıştırıldığı klasöre)

Not:
- Bu dosyayı proje tarafında `urban-light-sleep-ml/data/raw/` altına koymak gerekir.

### C) 311 noktalarını grid’e saymak

Dosya: [src/data/process_311.py](../src/data/process_311.py)

Ne yapar?
- `grid_cells.geojson` ve `nyc_311_noise_night_2020_present.csv` okur.
- Noktaları GeoDataFrame’e çevirir.
- Spatial join ile her şikayeti bir hücreye atar.
- Hücre bazında:
  - `noise_night_count` (toplam şikayet sayısı)
  - `peak_hour` (en yoğun saat; yoksa -1)
  üretir.

Çıktı:
- `data/processed/grid_311_counts.csv`

### D) VIIRS rasterını grid’e özetlemek

Dosya: [src/data/extract_viirs.py](../src/data/extract_viirs.py)

Ne yapar?
- `grid_cells.geojson` ile `NYC_VIIRS_Night_Lights_2020_present.tif` okur.
- Her hücre için rasterı maskeler.
- 0 ve nodata değerlerini dışarıda bırakıp ortalama alır.

Çıktı:
- `data/processed/grid_viirs_light.csv` (`cell_id`, `night_light_avg`)

### E) Dataset birleştirme + label üretimi

Dosya: [src/data/merge_datasets.py](../src/data/merge_datasets.py)

Ne yapar?
- 311 sayımları + VIIRS ışığı `cell_id` üzerinden birleştirir.
- `night_light_avg` eksikse median ile doldurur.
- Label üretir:
  - `high_noise_risk = 1` eğer `noise_night_count` >= 80. persentil
- Ek kolon:
  - `noise_night_count_log1p = log(1 + noise_night_count)`

Çıktı:
- `data/processed/final_model_dataset.csv`

## 3) Modelleme (Baseline)

### A) Baseline eğitim + metrik

Dosya: [src/models/train_baseline.py](../src/models/train_baseline.py)

- Features:
  - `night_light_avg`
  - `centroid_lat`, `centroid_lon`
  - `peak_hour`
- Target:
  - `high_noise_risk`

Modeller:
- Logistic Regression (StandardScaler + LogisticRegression)
- Random Forest

Çıktı:
- Konsolda Accuracy / F1 / ROC-AUC + classification_report.

### B) Confusion matrix görselleri

Dosya: [src/models/plot_confusion_matrices.py](../src/models/plot_confusion_matrices.py)

Çıktı:
- `outputs/figures/confusion_matrices/cm_logistic_regression.png`
- `outputs/figures/confusion_matrices/cm_random_forest.png`

## 4) Analiz / Görselleştirme

### A) Işık quantile analizi (tablo)

Dosya: [src/analysis/light_quantile_analysis.py](../src/analysis/light_quantile_analysis.py)

Ne yapar?
- `night_light_avg` değerlerini 5 quantile’a böler.
- Her quantile için özet istatistik çıkarır.

Çıktı:
- `outputs/tables/light_quantile_summary.csv`

### B) Quantile grafiği

Dosya: [src/analysis/plot_light_quantile.py](../src/analysis/plot_light_quantile.py)

Çıktı:
- `outputs/figures/light_quantile_vs_noise_risk.png`

### C) Dağılım grafiği: ışık vs gürültü

Dosya: [src/analysis/plot_scatter_light_vs_noise.py](../src/analysis/plot_scatter_light_vs_noise.py)

Çıktı:
- `outputs/figures/scatter_light_vs_noise.png`

### D) ROC eğrileri

Dosya: [src/analysis/plot_roc_curve.py](../src/analysis/plot_roc_curve.py)

Çıktı:
- `outputs/figures/roc_curve.png`

### E) RF feature importance

Dosya: [src/analysis/plot_feature_importance_rf.py](../src/analysis/plot_feature_importance_rf.py)

Çıktı:
- `outputs/figures/rf_feature_importance.png`

### F) Sınıf dağılımı

Dosya: [src/analysis/plot_class_distribution.py](../src/analysis/plot_class_distribution.py)

Çıktı:
- `outputs/figures/class_distribution.png`

## 5) Dashboard

Dosya: [src/visualization/dashboard_app.py](../src/visualization/dashboard_app.py)

Ne yapar?
- `grid_cells.geojson` geometrilerini `final_model_dataset.csv` ile birleştirir.
- Streamlit sidebar ile filtreleme sağlar.
- Folium haritada centroid noktaları ile risk görselleştirir.

Çalıştırma:
- `streamlit run src/visualization/dashboard_app.py`

## 6) Önerilen Çalıştırma Sırası

Aşağıdaki sırayı izlemek, dosyaların birbirine bağımlılığını doğru yönetir:

1. Grid üret: `python src/data/create_grid.py`
2. 311 indir: `python download_311_filtered.py` → çıktıyı `data/raw/` altına koy
3. 311 işle: `python src/data/process_311.py`
4. VIIRS işle: `python src/data/extract_viirs.py`
5. Dataset merge: `python src/data/merge_datasets.py`
6. Baseline train: `python src/models/train_baseline.py`
7. Grafikler:
   - `python src/models/plot_confusion_matrices.py`
   - `python src/analysis/plot_roc_curve.py`
   - `python src/analysis/plot_feature_importance_rf.py`
   - `python src/analysis/plot_class_distribution.py`
   - `python src/analysis/plot_scatter_light_vs_noise.py`
   - `python src/analysis/light_quantile_analysis.py`
   - `python src/analysis/plot_light_quantile.py`

---

İstersen bir sonraki adım olarak: bu dokümanı “Rapor” formatına (giriş/veri/özellik/sonuçlar) daha akademik hale de getirebilirim.
