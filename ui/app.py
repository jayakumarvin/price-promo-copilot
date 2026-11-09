# ui/app.py
import hmac, json, os, subprocess, sys, time, urllib.request
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
PROG = ROOT / "data" / "progress.json"
USER, PASS = os.environ.get("APP_USER", "admin"), os.environ.get("APP_PASS", "admin")
st.set_page_config(page_title="Price & Promo Copilot", layout="wide")

def progress():
    try:
        return json.loads(PROG.read_text())
    except Exception:
        return None

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")

def ollama_has_model():
    """True when the Ollama server answers and the chosen model has been pulled."""
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=2) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
        return OLLAMA_MODEL in names or f"{OLLAMA_MODEL}:latest" in names
    except Exception:
        return False

def same(a, b):
    return hmac.compare_digest(a.encode(), b.encode())

if not st.session_state.get("auth"):
    st.title("Sign in")
    with st.form("login"):
        u = st.text_input("User")
        p = st.text_input("Password", type="password")
        go = st.form_submit_button("Sign in")
    if go:
        if same(u, USER) and same(p, PASS):
            st.session_state["auth"] = True
            st.rerun()
        st.error("Wrong user or password")
    st.stop()

st.title("Price & Promo Copilot")
with st.sidebar:
    st.header("Data pipeline")
    if st.button("Run data pipeline"):
        PROG.parent.mkdir(exist_ok=True)
        PROG.unlink(missing_ok=True)
        log = open(ROOT / "data" / "pipeline.log", "w")
        subprocess.Popen([sys.executable, "-m", "pipeline.run"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        time.sleep(1.5)
        st.session_state["ready"] = False
        st.rerun()
    if st.button("Sign out"):
        st.session_state.clear()
        st.rerun()
    st.header("Language model")
    st.caption(f"Local Ollama model: {OLLAMA_MODEL}")
    if ollama_has_model():
        st.success("Ollama is running")
    else:
        st.warning(f"Ollama is not ready. Start it and run: ollama pull {OLLAMA_MODEL}")

@st.fragment(run_every=2)
def panel():
    p = progress()
    if not p:
        st.info("The pipeline has not run yet. Click Run data pipeline in the sidebar.")
        return
    st.progress(p["pct"] / 100, text=f"{p['pct']}% · {p['msg']} (at {p['ts']})")
    if p["state"] == "error":
        st.error(p["msg"])
    if p["state"] == "done" and not st.session_state.get("ready"):
        st.session_state["ready"] = True
        st.rerun()

panel()

if st.session_state.get("ready"):
    from agent.graph import graph
    q = st.chat_input("Ask about price, promotion, elasticity or policy")
    if q and not ollama_has_model():
        st.error(f"Ollama is not running or the model is missing. Run: ollama pull {OLLAMA_MODEL}")
    elif q:
        st.chat_message("user").write(q)
        with st.chat_message("assistant"):
            box = st.status("Working on it", expanded=True)
            final = ""
            for upd in graph.stream({"question": q, "log": []}, stream_mode="updates"):
                for node, out in upd.items():
                    for line in out.get("log", []):
                        box.write(f"- {line}")
                    if node == "sql" and out.get("sql"):
                        box.code(out["sql"], language="sql")
                    if "answer" in out:
                        final = out["answer"]
            box.update(label="Done", state="complete", expanded=False)
            st.markdown(final)
else:
    st.caption("Questions unlock when the pipeline reaches 100%.")
