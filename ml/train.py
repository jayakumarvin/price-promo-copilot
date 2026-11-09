# ml/train.py   (run with: python ml/train.py)
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor
from sqlalchemy import create_engine

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
ADMIN_URL = os.environ.get("PG_ADMIN_URL", "postgresql+psycopg2:///promo")
MIN_ROWS = 30


def load():
    eng = create_engine(ADMIN_URL)
    df = pd.read_sql("SELECT * FROM fact_weekly_sales", eng)
    df["week_start"] = pd.to_datetime(df["week_start"])
    for c in ["units", "list_price", "net_price", "discount_pct", "revenue"]:
        df[c] = pd.to_numeric(df[c])
    return df


def fit_elasticities(df):
    """One log-log regression per SKU: ln(units) ~ ln(net_price) + store fixed effects."""
    rows = []
    for sku, g in df[(df["units"] > 0) & (df["net_price"] > 0)].groupby("sku_id"):
        if len(g) < MIN_ROWS:
            continue
        try:
            g = g.assign(lnq=np.log(g["units"]), lnp=np.log(g["net_price"]))
            if g["lnp"].nunique() < 2:
                continue
            fit = smf.ols("lnq ~ lnp + C(store_id)", data=g).fit()
            e, p = float(fit.params["lnp"]), float(fit.pvalues["lnp"])
            rows.append(dict(sku_id=int(sku), elasticity=e, std_err=float(fit.bse["lnp"]), p_value=p,
                             r2=float(fit.rsquared), n_rows=len(g), reliable=bool(e < 0 and p < 0.05)))
        except Exception as ex:
            print(f"skipped SKU {sku}: {ex}")
    return pd.DataFrame(rows)


def features(d):
    return pd.DataFrame({"sku_id": d["sku_id"], "store_id": d["store_id"],
                         "month": d["week_start"].dt.month,
                         "week": d["week_start"].dt.isocalendar().week.astype(int),
                         "list_price": d["list_price"]})


def new_model():
    return HistGradientBoostingRegressor(categorical_features=[0, 1], max_iter=200, random_state=0)


def main():
    MODELS.mkdir(exist_ok=True)
    df = load()
    print(f"Loaded {len(df):,} weekly rows")

    el = fit_elasticities(df)
    el.to_csv(MODELS / "elasticity.csv", index=False)
    print(f"Elasticities: {len(el)} SKUs fitted, {int(el['reliable'].sum()) if len(el) else 0} reliable")

    # Baseline: units we would expect without a promotion, learned from non-promoted weeks only.
    base = df[df["promo_flag"] == 0].copy()
    cut = pd.Timestamp(base["week_start"].quantile(0.75))
    tr, te = base[base["week_start"] <= cut], base[base["week_start"] > cut]
    metrics = {"holdout_cut": str(cut.date()), "train_rows": len(tr), "test_rows": len(te)}
    if len(tr) >= MIN_ROWS and len(te) > 0:
        m = new_model().fit(features(tr), np.log1p(tr["units"]))
        pred = np.expm1(m.predict(features(te))).clip(min=0)
        metrics["holdout_wape"] = float(np.abs(te["units"] - pred).sum() / max(te["units"].sum(), 1))
        print(f"Baseline holdout WAPE: {metrics['holdout_wape']:.3f} (cut {metrics['holdout_cut']})")

    final = new_model().fit(features(base), np.log1p(base["units"]))
    joblib.dump(final, MODELS / "baseline.joblib")

    pr = df[df["promo_flag"] == 1]
    if len(pr):
        expected = np.expm1(final.predict(features(pr))).clip(min=1)
        metrics["avg_promo_lift"] = float((pr["units"] / expected).mean())
        print(f"Average units in inferred promo weeks vs baseline: {metrics['avg_promo_lift']:.2f}x")
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Models saved to", MODELS)


if __name__ == "__main__":
    main()
