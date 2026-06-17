
import pandas as pd
df = pd.read_parquet("data/processed/air_quality_weather.parquet")
s = df[df["station"] == "SANTURCE"].sort_values("Date")
print("Total rows:", len(s))
print("Date range:", s["Date"].min().date(), "to", s["Date"].max().date())
print("PM10 non-null:", s["PM10"].notna().sum())
print("PM10 null:", s["PM10"].isna().sum())
print(s[["Date","NO2","PM10","PM2.5","SO2"]].tail(5).to_string(index=False))