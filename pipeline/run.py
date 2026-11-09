# pipeline/run.py   (run with: python -m pipeline.run)
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
PROG, RAW = ROOT / "data" / "progress.json", ROOT / "data" / "raw"
SLUG = os.environ.get("KAGGLE_DATASET", "mashlyn/online-retail-ii-uci")
ADMIN_URL = os.environ.get("PG_ADMIN_URL", "postgresql+psycopg2:///promo")

def say(pct, msg, state="running"):
    tmp = PROG.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(pct=pct, msg=msg, state=state, ts=time.strftime("%H:%M:%S"))))
    tmp.replace(PROG)
    print(f"[{pct:3d}%] {msg}", flush=True)

def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        msg = (r.stderr or r.stdout).strip() or "command failed"
        raise RuntimeError(msg.splitlines()[-1][:300])

def main():
    say(2, "Checking for the Kaggle file")
    RAW.mkdir(parents=True, exist_ok=True)
    if not list(RAW.glob("*.csv")):
        say(5, f"Downloading {SLUG} from Kaggle")
        run([str(Path(sys.executable).parent / "kaggle"), "datasets", "download", "-d", SLUG, "-p", str(RAW), "--unzip"])
    csvs = list(RAW.glob("*.csv"))
    if not csvs:
        raise RuntimeError("No CSV in data/raw; the dataset may ship another file type")
    src = max(csvs, key=lambda p: p.stat().st_size)

    say(15, f"Counting rows in {src.name}")
    total = max(sum(1 for _ in open(src, "rb")) - 1, 1)
    parts, names, done = [], {}, 0
    for ch in pd.read_csv(src, chunksize=100_000, encoding="latin-1", dtype=str):
        ch.columns = [c.strip().lower().replace(" ", "_") for c in ch.columns]
        ch = ch.rename(columns={"invoiceno": "invoice", "unitprice": "price"})
        done += len(ch)
        ch["quantity"] = pd.to_numeric(ch["quantity"], errors="coerce")
        ch["price"] = pd.to_numeric(ch["price"], errors="coerce")
        ch = ch[~ch["invoice"].str.startswith("C", na=False) & (ch["quantity"] > 0) & (ch["price"] > 0)
                & ch["stockcode"].str.match(r"^\d{5}", na=False)]
        d = pd.to_datetime(ch["invoicedate"], errors="coerce")
        ch = ch.assign(week_start=(d.dt.normalize() - pd.to_timedelta(d.dt.dayofweek, unit="D")).dt.date,
                       revenue=ch["quantity"] * ch["price"]).dropna(subset=["week_start"])
        names.update(ch.drop_duplicates("stockcode").set_index("stockcode")["description"].dropna().to_dict())
        parts.append(ch.groupby(["week_start", "country", "stockcode"], as_index=False)
                       .agg(units=("quantity", "sum"), revenue=("revenue", "sum")))
        say(15 + int(30 * done / total), f"Cleaning rows: {done:,} of about {total:,}")

    say(46, "Building the weekly price and promotion table")
    w = pd.concat(parts).groupby(["week_start", "country", "stockcode"], as_index=False).sum()
    w["net_price"] = w["revenue"] / w["units"]
    top_sku = sorted(w.groupby("stockcode")["revenue"].sum().nlargest(100).index)
    top_ctry = sorted(w.groupby("country")["revenue"].sum().nlargest(5).index)
    w = w[w["stockcode"].isin(top_sku) & w["country"].isin(top_ctry)].copy()
    typical = w.groupby("stockcode")["net_price"].transform(lambda s: s.quantile(0.75))
    w["list_price"] = np.maximum(typical, w["net_price"])
    w["discount_pct"] = (1 - w["net_price"] / w["list_price"]).round(2)
    w["promo_flag"] = (w["discount_pct"] >= 0.10).astype(int)
    w["promo_type"] = np.where(w["promo_flag"] == 1, "TPR", "None")
    sku_map = {s: i + 1 for i, s in enumerate(top_sku)}
    st_map = {c: i + 1 for i, c in enumerate(top_ctry)}
    w["sku_id"], w["store_id"] = w["stockcode"].map(sku_map), w["country"].map(st_map)
    w["units"] = w["units"].round().astype(int)
    for c in ["list_price", "net_price", "revenue"]:
        w[c] = w[c].round(2)

    prod = pd.DataFrame({"sku_id": list(sku_map.values()), "sku_name": [str(names.get(s, s))[:60] for s in sku_map],
                         "category": "General", "brand": "n/a"})
    prod["base_price"] = prod["sku_id"].map(w.groupby("sku_id")["list_price"].max())
    prod["unit_cost"] = (prod["sku_id"].map(w.groupby("sku_id")["net_price"].median()) * 0.6).round(2)
    prod = prod.dropna(subset=["base_price"])
    stores = pd.DataFrame({"store_id": list(st_map.values()), "store_name": list(st_map),
                           "region": list(st_map), "format": "Country"})
    fact = w[["week_start", "store_id", "sku_id", "units", "list_price", "net_price",
              "promo_flag", "promo_type", "discount_pct", "revenue"]]
    pe = fact[fact["promo_flag"] == 1]
    promos = pd.DataFrame({"week_start": pe["week_start"], "store_id": pe["store_id"], "sku_id": pe["sku_id"],
                           "promo_type": pe["promo_type"], "discount_pct": pe["discount_pct"],
                           "trade_spend": ((pe["list_price"] - pe["net_price"]) * pe["units"]).round(2)})

    say(60, "Creating database tables")
    eng = create_engine(ADMIN_URL)
    with eng.begin() as c:
        c.exec_driver_sql((ROOT / "data" / "schema_kaggle.sql").read_text())
    prod.to_sql("dim_product", eng, if_exists="append", index=False)
    stores.to_sql("dim_store", eng, if_exists="append", index=False)
    n = len(fact)
    for i in range(0, n, 10_000):
        fact.iloc[i:i + 10_000].to_sql("fact_weekly_sales", eng, if_exists="append", index=False,
                                       method="multi", chunksize=2_000)
        say(65 + int(15 * min(i + 10_000, n) / n), f"Loading sales rows: {min(i + 10_000, n):,} of {n:,}")
    promos.to_sql("promo_events", eng, if_exists="append", index=False, method="multi", chunksize=2_000)

    say(82, "Loading policy documents into the vector store")
    run([sys.executable, "agent/ingest_policies.py"])
    say(90, "Training elasticity and promotion models")
    run([sys.executable, "ml/train.py"])
    say(100, f"Data ready: {n:,} weekly rows, {len(promos):,} inferred promotions", "done")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        say(0, f"Failed: {e}", "error")
        sys.exit(1)
