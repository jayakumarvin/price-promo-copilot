# ml/tools.py  - helpers the agent calls: SKU lookup, elasticity lookup, promo what-if
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RO_URL = os.environ.get("PG_RO_URL", "postgresql+psycopg2://promo_ro:promo_ro_pw@localhost/promo")
_engine = create_engine(RO_URL)


def elasticity_table() -> pd.DataFrame:
    return pd.read_csv(MODELS / "elasticity.csv")


def find_sku(query: str, limit: int = 5) -> pd.DataFrame:
    """Find SKUs by numeric sku_id or by part of the product name."""
    with _engine.connect() as c:
        if query.strip().isdigit():
            q, p = "SELECT sku_id, sku_name FROM dim_product WHERE sku_id = :v", {"v": int(query)}
        else:
            q, p = ("SELECT sku_id, sku_name FROM dim_product WHERE sku_name ILIKE :v "
                    "ORDER BY sku_id LIMIT :n"), {"v": f"%{query.strip()}%", "n": limit}
        return pd.DataFrame(c.execute(text(q), p).fetchall(), columns=["sku_id", "sku_name"])


def simulate_promo(sku_id: int, discount_pct: float) -> dict:
    """What-if for a price cut on one SKU, using its fitted elasticity and recent non-promoted weeks."""
    d = discount_pct / 100 if discount_pct > 1 else discount_pct
    el = elasticity_table()
    row = el[el["sku_id"] == sku_id]
    if row.empty:
        return {"ok": False, "reason": f"No elasticity estimate for SKU {sku_id} (fewer than 30 rows or no price variation)."}
    e, reliable = float(row["elasticity"].iloc[0]), bool(row["reliable"].iloc[0])

    with _engine.connect() as c:
        wk = pd.DataFrame(c.execute(text(
            "SELECT week_start, SUM(units) AS units, SUM(revenue) / NULLIF(SUM(units), 0) AS price "
            "FROM fact_weekly_sales WHERE sku_id = :s AND promo_flag = 0 "
            "GROUP BY week_start ORDER BY week_start DESC LIMIT 26"), {"s": sku_id}).fetchall(),
            columns=["week_start", "units", "price"])
        prod = c.execute(text("SELECT sku_name, unit_cost FROM dim_product WHERE sku_id = :s"), {"s": sku_id}).one()
    if wk.empty:
        return {"ok": False, "reason": f"SKU {sku_id} has no non-promoted weeks to use as a baseline."}

    units0, price0, cost = float(wk["units"].mean()), float(wk["price"].astype(float).mean()), float(prod.unit_cost)
    out = dict(ok=True, sku_id=sku_id, sku_name=prod.sku_name, discount_pct=round(d * 100, 1),
               elasticity=round(e, 2), elasticity_reliable=reliable,
               baseline_weekly_units=round(units0), baseline_price=round(price0, 2), unit_cost_assumed=round(cost, 2))
    if e >= 0:
        out.update(ok=False, reason=f"Estimated elasticity is {e:.2f} (not negative), so a lift forecast would be meaningless.")
        return out

    price1 = price0 * (1 - d)
    units1 = units0 * (1 - d) ** e  # constant-elasticity demand curve
    out.update(promo_price=round(price1, 2), promo_weekly_units=round(units1),
               unit_lift_pct=round((units1 / units0 - 1) * 100, 1),
               baseline_margin=round((price0 - cost) * units0), promo_margin=round((price1 - cost) * units1),
               discount_cost=round((price0 - price1) * units1),
               promo_margin_pct=round((price1 - cost) / price1 * 100, 1))
    out["incremental_margin"] = out["promo_margin"] - out["baseline_margin"]
    return out
