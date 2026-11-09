# agent/cli.py   (run with: python -m agent.cli "your question")
import sys

from agent.graph import graph


def ask(q: str) -> str:
    final = ""
    for upd in graph.stream({"question": q, "log": []}, stream_mode="updates"):
        for node, out in upd.items():
            for line in out.get("log", []):
                print(f"  - {line}", flush=True)
            if node == "sql" and out.get("sql"):
                print(f"    SQL: {out['sql']}", flush=True)
            if "answer" in out:
                final = out["answer"]
    print(f"\n{final}\n")
    return final


def main():
    if len(sys.argv) > 1:
        for q in sys.argv[1:]:
            print(f"Q: {q}")
            ask(q)
        return
    while True:
        try:
            q = input("Question (blank to quit): ").strip()
        except EOFError:
            break
        if not q:
            break
        ask(q)


if __name__ == "__main__":
    main()
