
import inspect, pathlib, circuitkit as ck
root = pathlib.Path(ck.__file__).parent
print("files mentioning chat_template:", [str(p.relative_to(root)) for p in root.rglob("*.py") if "chat_template" in p.read_text(errors="ignore")][:8])
try:
    t = ck.get_task("multilingual_refusal"); print("task object:", type(t), "| file:", inspect.getsourcefile(type(t)))
    print(inspect.getsource(type(t))[:2500])
except Exception as e: print("task inspect failed:", repr(e)[:200])
