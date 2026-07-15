#!/usr/bin/env python3
"""Regression guard: gpu_demosaic_pytorch_unpacked's return_linear kwarg
must actually reach the code that uses it.

Real bug (from a live run's log): gpu_demosaic_pytorch_unpacked accepted
return_linear but called _gpu_demosaic_pytorch_body(unpacked, device,
cancel_check) without it -- and _gpu_demosaic_pytorch_body didn't declare
the parameter at all, only referenced the name `if return_linear:` in its
body. Every GPU (pytorch_mps/cuda) decode raised NameError and silently
fell back to CPU decode.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    import gpu_raw_processor as g

    sig = inspect.signature(g._gpu_demosaic_pytorch_body)
    check(
        "_gpu_demosaic_pytorch_body declares return_linear",
        "return_linear" in sig.parameters,
    )

    wrapper_src = inspect.getsource(g.gpu_demosaic_pytorch_unpacked)
    import ast
    tree = ast.parse(wrapper_src)
    passed = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_gpu_demosaic_pytorch_body":
            args_names = [getattr(arg, "id", None) for arg in node.args]
            kw_names = [kw.arg for kw in node.keywords]
            if "return_linear" in args_names or "return_linear" in kw_names:
                passed = True
                break
    check(
        "gpu_demosaic_pytorch_unpacked forwards return_linear to the body call",
        passed,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
