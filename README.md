# Urban Light Pollution and Nighttime Noise Risk Analysis (NYC)

This project investigates the relationship between **nighttime light intensity** and **nighttime noise complaints** in New York City using **machine learning and spatial analysis** techniques.

By integrating **satellite-based night light data (VIIRS)** with **NYC 311 noise complaint records**, the study aims to identify urban areas with elevated nighttime noise risk and to evaluate whether artificial light intensity can serve as a meaningful proxy indicator for urban noise-related disturbances.

---

## 📌 Project Motivation

Urban environmental problems such as **noise pollution** and **light pollution** are known to negatively affect public health, sleep quality, and overall quality of life.  
However, these phenomena are often analyzed independently.

This project addresses the following research questions:

- Is there a measurable relationship between nighttime light intensity and nighttime noise complaints?
- Can satellite-based night light data help identify high-risk noise zones in urban environments?
- How effectively can machine learning models classify high noise risk areas using spatial and environmental features?

---

## 📂 Data Sources

### 1️⃣ NYC 311 Noise Complaints  
- Source: NYC Open Data (Socrata API)  
- Time range: **2020 – present**  
- Filter: Nighttime residential noise complaints  
- Attributes: timestamp, location (latitude/longitude), complaint type

### 2️⃣ VIIRS Nighttime Lights  
- Source: NASA / NOAA (Google Earth Engine catalog)  
- Product: Monthly averaged night light radiance  
- Coverage: New York City  
- Resolution: Satellite raster data

### 3️⃣ NYC Borough Boundaries  
- Format: GeoJSON  
- Purpose: Spatial clipping and grid generation

---

## 🧠 Methodology (CRISP-DM)

The project follows the **CRISP-DM** data mining methodology:

1. **Business Understanding**  
   - Identify high-risk nighttime noise areas in NYC.
2. **Data Understanding**  
   - Explore spatial distribution of noise complaints and light intensity.
3. **Data Preparation**  
   - Generate a spatial grid over NYC  
   - Aggregate noise complaints per grid cell  
   - Extract average night light intensity per cell
4. **Modeling**  
   - Binary classification of high-risk vs low-risk noise zones
5. **Evaluation**  
   - Accuracy, F1-score, ROC-AUC, confusion matrices
6. **Interpretation**  
   - Feature importance and environmental implications

---

## 🗺️ Spatial Processing Pipeline

1. Create a **fishnet grid** covering NYC
2. Assign each 311 complaint to a grid cell (spatial join)
3. Aggregate complaint counts per grid cell
4. Extract VIIRS night light values for each grid cell
5. Merge datasets into a final modeling table

---

## 🤖 Machine Learning Models

Two baseline classifiers are implemented:

### 🔹 Logistic Regression
- Interpretable linear baseline
- Provides probability estimates

### 🔹 Random Forest
- Handles nonlinear relationships
- Provides feature importance scores

### 🎯 Target Variable
- `high_noise_risk`
- Defined as grid cells above the **80th percentile** of nighttime noise complaints

---

## 📊 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix

Random Forest achieved superior performance, especially in identifying high-risk noise areas.

---

## 📈 Key Findings

- Nighttime light intensity shows a **strong positive association** with nighttime noise complaints.
- High light intensity zones are significantly more likely to be classified as high noise risk.
- Spatial and environmental proxies can effectively support urban noise analysis.
- Machine learning models, particularly Random Forest, perform well in identifying high-risk zones.

---

## 🧪 Project Structure

urban-light-sleep-ml/
│
├── data/
│ ├── raw/ # Original datasets (311, VIIRS, GeoJSON)
│ └── processed/ # Grid-based aggregated datasets
│
├── src/
│ ├── data/ # Data processing scripts
│ ├── models/ # ML training and evaluation
│ ├── analysis/ # Statistical and exploratory analysis
│ └── visualization/ # Plots and dashboard code
│
├── outputs/
│ ├── figures/ # Generated plots
│ └── tables/ # Summary tables
│
├── app.py # Dashboard application
└── README.md


---

## 🧑‍🏫 Academic Relevance

This project demonstrates:
- Integration of **remote sensing data** with urban open data
- Application of **machine learning** to real-world environmental problems
- Use of **spatial aggregation and geospatial analysis**
- Clear alignment with **CRISP-DM** methodology

---

## 🚀 Future Work

- Extend analysis to other cities
- Incorporate additional environmental variables (traffic, land use)
- Use temporal modeling (time series)
- Deploy real-time dashboards for urban monitoring

---

## 📜 License

This project is intended for **academic and educational use**.  
Original datasets are subject to their respective open data licenses.


