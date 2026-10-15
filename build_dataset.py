
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

RAW = "data/raw/listings.csv.gz"
OUT = "data/processed/listings_scored.csv.gz"
CENTRO = (51.5074, -0.1278)  # Centro de londres

FLAGS = {
    "has_ac": r"air conditioning",
    "has_elevator": r"elevator",
    "has_pool": r"\bpool\b(?! table)",  
    "has_parking": r"parking",
    "has_hot_tub": r"hot tub|jacuzzi",
    "has_gym": r"\bgym\b",
    "has_washer": r"washer",
}
NUM = [
    "accommodates", "bedrooms", "beds", "bathrooms", "latitude", "longitude",
    "distance_center_km", "amenities_count", "minimum_nights", "number_of_reviews",
    "review_scores_rating", "review_scores_location", "availability_365", "host_years",
] + list(FLAGS)
CAT = ["room_type", "property_type", "neighbourhood_cleansed"]
OPCIONALES = ["id", "name", "listing_url", "bathrooms", "beds", "host_since", "minimum_nights",
              "number_of_reviews", "review_scores_rating", "review_scores_location",
              "availability_365", "amenities", "property_type"]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def n_amenities(s):
    try:
        return len(json.loads(s))
    except Exception:
        return s.count(",") + 1


def main():
    df = pd.read_csv(RAW, low_memory=False)
    for c in OPCIONALES:
        if c not in df.columns:
            df[c] = np.nan

    df["price"] = pd.to_numeric(df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    df = df[df["room_type"].isin(["Entire home/apt", "Hotel room"])]
    df = df.dropna(subset=["price", "latitude", "longitude", "accommodates"])
    df = df[df["price"] > 0]
    lo, hi = df["price"].quantile([0.01, 0.99])
    df = df[df["price"].between(lo, hi)].copy()

    txt = df["bathrooms_text"].astype(str)
    parsed = txt.str.extract(r"(\d+\.?\d*)")[0].astype(float)
    parsed = parsed.mask(txt.str.contains("half", case=False) & parsed.isna(), 0.5)
    df["bathrooms"] = df["bathrooms"].fillna(parsed)

    df["distance_center_km"] = haversine_km(df["latitude"], df["longitude"], *CENTRO)
    df["host_years"] = (pd.Timestamp.today() - pd.to_datetime(df["host_since"], errors="coerce")).dt.days / 365.25
    am = df["amenities"].fillna("[]").astype(str)
    df["amenities_count"] = am.map(n_amenities)
    low = am.str.lower()
    for k, pat in FLAGS.items():
        df[k] = low.str.contains(pat, regex=True).astype(int)
    for c in CAT:
        df[c + "_code"] = df[c].astype("category").cat.codes
    feats = NUM + [c + "_code" for c in CAT]

    y = np.log(df["price"])
    model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, random_state=42, n_jobs=-1)
    pred_log = cross_val_predict(model, df[feats], y, cv=KFold(5, shuffle=True, random_state=42))
    df["pred_price"] = np.exp(pred_log)

    print(f"Alojamientos: {len(df):,}")
    print(f"MAE (£): {mean_absolute_error(df['price'], df['pred_price']):.1f}")
    print(f"R² (log precio): {r2_score(y, pred_log):.3f}   <- si sale > 0.9, busca fugas")
    imp = pd.Series(model.fit(df[feats], y).feature_importances_, index=feats).nlargest(8)
    print("Variables más importantes:\n", imp.round(3).to_string())

    out = df[["id", "name", "listing_url", "neighbourhood_cleansed", "room_type", "property_type",
              "latitude", "longitude", "accommodates", "bedrooms", "bathrooms", "amenities_count",
              "distance_center_km", "review_scores_rating", "number_of_reviews", "price", "pred_price"]]
    out = out.rename(columns={"neighbourhood_cleansed": "borough"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, compression="gzip", float_format="%.4f")
    print(f"Guardado en {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()