"""download_311_filtered.py

Bu script, NYC Open Data (Socrata) üzerinden 311 kayıtlarını indirip gece saatlerine filtreleyerek CSV olarak kaydeder.

Amaç
- Dataset: `erm2-nwe9` (311 Service Requests)
- Şimdilik sadece `Noise - Residential` kayıtlarını çekmek.
- 2020-01-01 ve sonrası kayıtlar.
- Gece filtresi: 22:00–06:00 (proje kapsamındaki "gece gürültüsü" tanımı)

Çıktı
- Çalıştırdığın dizine: nyc_311_noise_night_2020_present.csv
    (Bu dosyayı proje tarafında `urban-light-sleep-ml/data/raw/` altına taşıyıp
     `src/data/process_311.py` ile grid sayımı üretiyoruz.)

Notlar / sınırlamalar
- App token kullanılmadığı için (client = Socrata(..., None)) istekler daha yavaş olabilir
    ve rate limit'e takılma ihtimali artar.
- İndirme pagination ile yapılır (LIMIT/OFFSET).
"""

import pandas as pd
from sodapy import Socrata

# App token yok: olur ama yavaş (şimdilik yeterli). Rate limit sorun olursa token eklemek gerekir.
client = Socrata("data.cityofnewyork.us", None)

SELECT = "created_date, complaint_type, latitude, longitude, borough"

WHERE = """
complaint_type = 'Noise - Residential'
AND created_date >= '2020-01-01T00:00:00.000'
AND latitude IS NOT NULL
"""

LIMIT = 50000
offset = 0
all_rows = []

print("Downloading 311 data in chunks...")

while True:
    # Socrata API: select/where/limit/offset ile sayfalama
    results = client.get(
        "erm2-nwe9",
        select=SELECT,
        where=WHERE,
        limit=LIMIT,
        offset=offset
    )

    if not results:
        break

    all_rows.extend(results)
    offset += LIMIT
    print(f"Downloaded {len(all_rows)} rows...")

df = pd.DataFrame.from_records(all_rows)

# created_date alanını datetime'a çevir
df["created_date"] = pd.to_datetime(df["created_date"])

# Gece filtresi: 22:00–06:00
df = df[(df["created_date"].dt.hour >= 22) | (df["created_date"].dt.hour <= 6)]

df.to_csv("nyc_311_noise_night_2020_present.csv", index=False)
print("Saved nyc_311_noise_night_2020_present.csv")
