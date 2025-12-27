import pandas as pd
from sodapy import Socrata

# App token yok, olur ama yavaş (şimdilik yeterli)
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

# Datetime
df["created_date"] = pd.to_datetime(df["created_date"])

# Gece filtresi: 22:00–06:00
df = df[(df["created_date"].dt.hour >= 22) | (df["created_date"].dt.hour <= 6)]

df.to_csv("nyc_311_noise_night_2020_present.csv", index=False)
print("Saved nyc_311_noise_night_2020_present.csv")
