import numpy as np
import pandas as pd
from scipy.special import expit, logit

ms = [
    {"rating": "Aaa", "upper": 0.0005, "pd_mid": 0.0003},
    {"rating": "Aa1", "upper": 0.0010, "pd_mid": 0.0007},
    {"rating": "Aa2", "upper": 0.0017, "pd_mid": 0.0013},
    {"rating": "Aa3", "upper": 0.0043, "pd_mid": 0.0028},
    {"rating": "A1", "upper": 0.0077, "pd_mid": 0.0058},
    {"rating": "A2", "upper": 0.0102, "pd_mid": 0.0089},
    {"rating": "A3", "upper": 0.0175, "pd_mid": 0.0135},
    {"rating": "B1", "upper": 0.0235, "pd_mid": 0.0204},
    {"rating": "B2", "upper": 0.0370, "pd_mid": 0.0297},
    {"rating": "B3", "upper": 0.0490, "pd_mid": 0.0427},
    {"rating": "C1", "upper": 0.0643, "pd_mid": 0.0563},
    {"rating": "C2", "upper": 0.0800, "pd_mid": 0.0719},
    {"rating": "C3", "upper": 0.1100, "pd_mid": 0.0942},
    {"rating": "D1", "upper": 0.1500, "pd_mid": 0.129},
    {"rating": "D2", "upper": 0.4000, "pd_mid": 0.2549},
    {"rating": "D3", "upper": 0.7000, "pd_mid": 0.6},
    {"rating": "D4", "upper": 1, "pd_mid": 0.813},
]

def pd_to_rating(df, ms, col_pd, col_rating):
    bins = [0.0] + [row["upper"] for row in ms]
    labels = [row["rating"] for row in ms]
    
    df["col_rating"] = pd.cut(df[col_pd], bins=bins, labels= labels, right=True)
    df["col_rating"] = df["col_rating"].astype("object")
    
    return df


def rating_to_pd_mid(df, ms, col_rating, col_pd_mid):
    rating_to_pd_mid_map = {row["rating"]: row["pd_mid"] for row in ms}
    
    df[col_pd_mid] = df[col_rating].map(rating_to_pd_mid_map.astype(float))
    
    return df