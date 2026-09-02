import numpy as np
import pandas as pd
from scipy.special import expit, logit

ms = [
    {"rating": "Aaa", "upper": 0.0005, "pd_mid": 0.0003, "upper_score": 1000},
    {"rating": "Aa1", "upper": 0.0010, "pd_mid": 0.0007, "upper_score": 750},
    {"rating": "Aa2", "upper": 0.0017, "pd_mid": 0.0013, "upper_score": 710},
    {"rating": "Aa3", "upper": 0.0043, "pd_mid": 0.0028, "upper_score": 680},
    {"rating": "A1", "upper": 0.0077, "pd_mid": 0.0058, "upper_score": 661},
    {"rating": "A2", "upper": 0.0102, "pd_mid": 0.0089, "upper_score": 645},
    {"rating": "A3", "upper": 0.0175, "pd_mid": 0.0135, "upper_score": 630},
    {"rating": "B1", "upper": 0.0235, "pd_mid": 0.0204, "upper_score": 615},
    {"rating": "B2", "upper": 0.0370, "pd_mid": 0.0297, "upper_score": 600},
    {"rating": "B3", "upper": 0.0490, "pd_mid": 0.0427, "upper_score": 591},
    {"rating": "C1", "upper": 0.0643, "pd_mid": 0.0563, "upper_score": 582},
    {"rating": "C2", "upper": 0.0800, "pd_mid": 0.0719, "upper_score": 571},
    {"rating": "C3", "upper": 0.1100, "pd_mid": 0.0942, "upper_score": 560},
    {"rating": "D1", "upper": 0.1500, "pd_mid": 0.129, "upper_score": 552},
    {"rating": "D2", "upper": 0.4000, "pd_mid": 0.2549, "upper_score": 525},
    {"rating": "D3", "upper": 0.7000, "pd_mid": 0.6, "upper_score": 460},
    {"rating": "D4", "upper": 1, "pd_mid": 0.813, "upper_score": 335},
]

#Right score true (750 < Aaa <= 1000)

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

# ── 3. Score → Rating ────────────────────────────────────────────────────────
def score_to_rating(df, ms, col_score, col_rating):
    """
    Quy đổi Score → Rating

    Ví dụ:
        750 < Score <= 1000 → Aaa
        710 < Score <= 750  → Aa1
        ...
        335 < Score <= 460  → D3
        Score <= 335        → D4
    """

    # upper_score tăng dần
    ms_sorted = sorted(ms, key=lambda x: x["upper_score"])

    bins = [-np.inf] + [
        row["upper_score"]
        for row in ms_sorted
    ]

    labels = [
        row["rating"]
        for row in ms_sorted
    ]

    df[col_rating] = pd.cut(
        df[col_score],
        bins=bins,
        labels=labels,
        right=True
    ).astype("object")

    return df


# ── 4. PD → Score ────────────────────────────────────────────────────────────
def pd_to_score(
    df,
    col_pd,
    col_score,
    base_score=600,
    base_odds=50,
    pdo=25
):
    """
    Score = Offset + Factor * ln(Odds)

    Với:
        Odds = Good / Bad = (1 - PD) / PD

        Factor = PDO / ln(2)

        Offset = BaseScore - Factor * ln(BaseOdds)
    """

    factor = pdo / np.log(2)

    offset = (
        base_score
        - factor * np.log(base_odds)
    )

    pd_values = df[col_pd].clip(1e-10, 1 - 1e-10)

    odds = (1 - pd_values) / pd_values

    df[col_score] = (
        offset
        + factor * np.log(odds)
    )

    return df