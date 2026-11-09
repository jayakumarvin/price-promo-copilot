# agent/graph.py  - LangGraph agent: plan -> sql -> policy -> simulate -> answer
import os
import sys
from pathlib import Path
from typing import Optional, TypedDict

import pandas as pd
import sqlglot
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ml import tools as mlt  # noqa: E402

load_dotenv()
ENGINE = create_engine(os.environ.get("PG_RO_URL", "postgresql+psycopg2://promo_ro:promo_ro_pw@localhost/promo"))
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
# Free local model. Weaker than a hosted model at SQL and at following policy text, so the
# SQL step validates every query and retries once with the database error.
LLM = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0, num_ctx=8192)
ALLOWED_TABLES = {"dim_product", "dim_store", "fact_weekly_sales", "promo_events"}


def _weeks() -> str:
    try:
        with ENGINE.connect() as c:
            lo, hi = c.execute(text("SELECT MIN(week_start), MAX(week_start) FROM fact_weekly_sales")).one()
        return f"{lo} to {hi}"
    except Exception:
        return "unknown (could not read fact_weekly_sales)"


SCHEMA = f"""PostgreSQL tables (read only):
dim_product(sku_id, sku_name, category, brand, base_price, unit_cost)  -- category is always 'General', brand 'n/a'
dim_store(store_id, store_name, region, format)  -- each "store" is a country
fact_weekly_sales(week_start, store_id, sku_id, units, list_price, net_price, promo_flag, promo_type, discount_pct, revenue)
promo_events(event_id, week_start, store_id, sku_id, promo_type, discount_pct, trade_spend)
week_start is a Monday. Weekly data covers {_weeks()}.
promo_type is None or TPR. TPR means the weekly net price was at least 10% below the SKU's typical list price
(the 75th percentile of its weekly net price). This is inferred from prices, not a real promotion calendar.
unit_cost is not in the source data: it is assumed to be 60% of the SKU's median net price.
"""

SYSTEM = f"""You are a pricing and promotion analyst for a retailer. Answer from the data, model results and policy
excerpts you are given, and never invent numbers.
{SCHEMA}
Important limits:
- Promotions are inferred from price drops, not recorded. Unit cost is assumed at 60% of median price.
- Whenever an answer relies on promotion flags, trade spend, margin or unit cost, say plainly that it depends on these assumptions.
- Elasticities are fitted per SKU from this data. Say when an estimate is marked unreliable.
- Quote policy limits from the excerpts and name the source file. If no excerpt covers the point, say so.
Keep answers short and concrete, with the key numbers first."""


class Sim(BaseModel):
    sku: str = Field(description="SKU id or part of the product name")
    discount_pct: float = Field(description="Discount in percent, e.g. 15 for 15% off")


class Plan(BaseModel):
    use_sql: bool = Field(description="Needs numbers from the database")
    use_policy: bool = Field(description="Needs company pricing or promotion policy")
    simulate: Optional[Sim] = Field(default=None, description="Set only for a what-if promo on one product")


class SqlOut(BaseModel):
    sql: str = Field(description="One PostgreSQL SELECT statement")


class State(TypedDict, total=False):
    question: str
    log: list
    plan: dict
    sql: str
    rows: str
    policy: str
    sim: dict
    answer: str


def _text(msg) -> str:
    c = msg.content
    return c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))


def check_sql(sql: str) -> str:
    stmts = sqlglot.parse(sql, read="postgres")
    if len(stmts) != 1 or stmts[0] is None:
        raise ValueError("Exactly one statement is allowed")
    tree = stmts[0]
    if not isinstance(tree, (exp.Select, exp.Union)):
        raise ValueError("Only SELECT statements are allowed")
    if tree.find(exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command):
        raise ValueError("Only read queries are allowed")
    ctes = {c.alias for c in tree.find_all(exp.CTE)}
    bad = {t.name for t in tree.find_all(exp.Table)} - ALLOWED_TABLES - ctes
    if bad:
        raise ValueError(f"Unknown tables: {sorted(bad)}")
    return tree.sql(dialect="postgres")


def run_sql(sql: str) -> pd.DataFrame:
    safe = f"SELECT * FROM ({sql}) AS q LIMIT 200".replace(":", "\\:")
    with ENGINE.connect() as c:
        c.execute(text("SET statement_timeout = 15000"))
        res = c.execute(text(safe))
        return pd.DataFrame(res.fetchall(), columns=list(res.keys()))


def plan_node(s: State):
    plan = LLM.with_structured_output(Plan).invoke([
        SystemMessage(content=SCHEMA + "\nDecide which sources are needed for the question. "
                      "Use simulate only for what-if questions about cutting the price of one product."),
        HumanMessage(content=s["question"])])
    parts = [n for n, on in [("data", plan.use_sql), ("policy", plan.use_policy), ("simulation", plan.simulate)] if on]
    return {"plan": plan.model_dump(), "log": ["Plan: use " + (", ".join(parts) or "general knowledge only")]}


def sql_node(s: State):
    if not s["plan"]["use_sql"]:
        return {"log": []}
    err, log = "", []
    for attempt in range(2):
        prompt = s["question"] + (f"\nYour previous query failed: {err}\nFix it." if err else "")
        sql = LLM.with_structured_output(SqlOut).invoke([
            SystemMessage(content=SCHEMA + "\nWrite one PostgreSQL SELECT that answers the question. "
                          "Use only these tables. Keep results small (aggregate, top N)."),
            HumanMessage(content=prompt)]).sql.strip().rstrip(";")
        try:
            sql = check_sql(sql)
            df = run_sql(sql)
            return {"sql": sql, "rows": df.to_string(max_rows=60, index=False),
                    "log": log + [f"Query returned {len(df)} rows"]}
        except Exception as e:
            err = str(e).splitlines()[0][:300]
            log.append(f"Query attempt {attempt + 1} failed: {err}")
    return {"rows": f"(query failed: {err})", "log": log}


_store = None


def policy_node(s: State):
    global _store
    if not s["plan"]["use_policy"]:
        return {"log": []}
    if _store is None:
        _store = Chroma(collection_name="policies", persist_directory=str(ROOT / "chroma_db"),
                        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    docs = _store.similarity_search(s["question"], k=4)
    body = "\n\n".join(f"[{d.metadata.get('source')}]\n{d.page_content}" for d in docs)
    return {"policy": body, "log": [f"Found {len(docs)} policy excerpts"]}


def simulate_node(s: State):
    sim = s["plan"].get("simulate")
    if not sim:
        return {"log": []}
    cand = mlt.find_sku(sim["sku"], limit=1)
    if cand.empty:
        return {"sim": {"ok": False, "reason": f"No product matches '{sim['sku']}'"}, "log": ["No matching product"]}
    sku = int(cand["sku_id"].iloc[0])
    res = mlt.simulate_promo(sku, float(sim["discount_pct"]))
    return {"sim": res, "log": [f"Simulated {sim['discount_pct']}% off SKU {sku} ({cand['sku_name'].iloc[0]})"]}


def answer_node(s: State):
    ctx = [f"Question: {s['question']}"]
    if s.get("sql"):
        ctx.append(f"SQL run:\n{s['sql']}\n\nResult:\n{s.get('rows', '')}")
    elif s.get("rows"):
        ctx.append(s["rows"])
    if s.get("policy"):
        ctx.append("Policy excerpts:\n" + s["policy"])
    if s.get("sim"):
        ctx.append(f"Promo simulation (constant-elasticity model): {s['sim']}")
    msg = LLM.invoke([SystemMessage(content=SYSTEM), HumanMessage(content="\n\n".join(ctx))])
    return {"answer": _text(msg), "log": ["Answer written"]}


def build():
    g = StateGraph(State)
    for name, fn in [("plan", plan_node), ("sql", sql_node), ("policy", policy_node),
                     ("simulate", simulate_node), ("answer", answer_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "sql")
    g.add_edge("sql", "policy")
    g.add_edge("policy", "simulate")
    g.add_edge("simulate", "answer")
    g.add_edge("answer", END)
    return g.compile()


graph = build()
