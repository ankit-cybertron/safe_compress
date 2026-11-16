
import sys, inspect
import circuitkit as ck
print("circuitkit", getattr(ck, "__version__", "?")); print([n for n in dir(ck) if not n.startswith("_")])
for fn in ("load_model", "discover", "evaluate", "export_checkpoint"):
    if hasattr(ck, fn):
        f = getattr(ck, fn); print("\n--- " + fn + str(inspect.signature(f))); print((f.__doc__ or "no docstring")[:1500])
if len(sys.argv) > 1 and sys.argv[1] == "run":
    cm = ck.load_model("CohereLabs/tiny-aya-global"); circuit = ck.discover(cm, "ioi", n_examples=32)
    print(type(circuit)); print([n for n in dir(circuit) if not n.startswith("_")][:60])
