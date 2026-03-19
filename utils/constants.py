AIRLINE_NAMES = {
    "AA": "American", "DL": "Delta",     "WN": "Southwest", "UA": "United",
    "AS": "Alaska",   "B6": "JetBlue",   "NK": "Spirit",    "F9": "Frontier",
    "HA": "Hawaiian", "G4": "Allegiant", "OO": "SkyWest",   "9E": "Endeavor",
    "MQ": "Envoy",    "YX": "Republic",  "OH": "PSA",       "QX": "Horizon",
    "YV": "Mesa",
}

AIRLINE_CODES = sorted(AIRLINE_NAMES.keys())

def airline_label(code):
    return f"{AIRLINE_NAMES.get(code, code)} ({code})"

MODEL_PATHS = {
    "classifier":   "models/lgbm_classifier.joblib",
    "regressor":    "models/lgbm_regressor.joblib",
    "encoder":      "models/ordinal_encoder.joblib",
    "top_orig":     "models/top_orig.joblib",
    "top_dest":     "models/top_dest.joblib",
    "route_data":   "models/route_avg_delay.joblib",
    "preprocessor": "models/preprocessor_sample.joblib",
    "insights":     "models/insights_stats.joblib",
}
