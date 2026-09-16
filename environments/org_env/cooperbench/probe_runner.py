"""Fixed, stdlib-only entry point copied into the public task sandbox.

This module is never executed against candidate code on the host. Each check
gets fresh source, process state and fixture data; migration setup runs using
the untouched public baseline, not the candidate's new writer.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def is_assertion_guard(node):
    """A dynamic condition controlling an explicit AssertionError failure."""
    if not isinstance(node, ast.If) or isinstance(node.test, ast.Constant):
        return False
    def contains(nodes):
        for child in nodes:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Raise):
                exc = child.exc
                if isinstance(exc, ast.Call):
                    exc = exc.func
                if isinstance(exc, ast.Name) and exc.id == "AssertionError":
                    return True
            if contains(ast.iter_child_nodes(child)):
                return True
        return False
    return contains(node.body + node.orelse)


CHILD = r'''
import ast, json, os, sys, traceback, threading
from pathlib import Path
source = Path(sys.argv[1]).read_text(encoding="utf-8")
repo = Path.cwd().resolve()
expected_paths = set(json.loads(sys.argv[2]))
stats = {"assertions": 0, "assertion_guards": 0, "public_calls": 0,
         "public_accesses": 0}
public_paths = set()
public_reads = set()
public_bindings = {}
def path_for_module(module):
    candidates = (module.replace(".", "/") + ".py",
                  module.replace(".", "/") + "/__init__.py")
    for expected in expected_paths:
        if any(expected == candidate or expected.endswith("/" + candidate)
               for candidate in candidates):
            return expected
    return None
def collect_public_bindings(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            path = path_for_module(node.module)
            for alias in node.names:
                if alias.name == "*":
                    continue
                binding_path = path or path_for_module(
                    node.module + "." + alias.name
                )
                if binding_path:
                    public_bindings[alias.asname or alias.name] = binding_path
def audit(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = Path(os.fsdecode(args[0]))
        if not path.is_absolute():
            path = repo / path
        try:
            relative = path.resolve().relative_to(repo).as_posix()
            if relative in expected_paths:
                public_reads.add(relative)
        except ValueError:
            pass
sys.addaudithook(audit)
def checked(value, message=None):
    stats["assertions"] += 1
    if not value:
        raise AssertionError(message() if message else "public behavior assertion failed")
def guarded(value):
    stats["assertion_guards"] += 1
    return value
def public_access(path, value):
    stats["public_accesses"] += 1
    public_paths.add(path)
    return value
def is_assertion_guard(node):
    if not isinstance(node, ast.If) or isinstance(node.test, ast.Constant):
        return False
    def contains(nodes):
        for child in nodes:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Raise):
                exc = child.exc
                if isinstance(exc, ast.Call):
                    exc = exc.func
                if isinstance(exc, ast.Name) and exc.id == "AssertionError":
                    return True
            if contains(ast.iter_child_nodes(child)):
                return True
        return False
    return contains(node.body + node.orelse)
class Instrument(ast.NodeTransformer):
    def visit_Name(self, node):
        path = public_bindings.get(node.id) if isinstance(node.ctx, ast.Load) else None
        if path:
            return ast.copy_location(ast.Call(
                func=ast.Name(id="_cooper_public_access", ctx=ast.Load()),
                args=[ast.Constant(path), node], keywords=[]), node)
        return node
    def visit_Assert(self, node):
        # A model may append ``assert True`` after a call merely to express
        # "did not raise". Successful execution of the call is useful, but the
        # constant itself is not an assertion and must not inflate evidence.
        if isinstance(node.test, ast.Constant) and bool(node.test.value):
            return ast.copy_location(ast.Pass(), node)
        message = ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                             body=self.visit(node.msg)) if node.msg else ast.Constant(None)
        return ast.copy_location(ast.Expr(ast.Call(
            func=ast.Name(id="_cooper_checked", ctx=ast.Load()),
            args=[self.visit(node.test), message],
            keywords=[])), node)
    def visit_If(self, node):
        negative_check = is_assertion_guard(node)
        self.generic_visit(node)
        if negative_check:
            node.test = ast.copy_location(ast.Call(func=ast.Name(id="_cooper_guarded", ctx=ast.Load()),
                                                   args=[node.test], keywords=[]), node.test)
        return node
def trace(frame, event, arg):
    if event == "call" and frame.f_code.co_name != "<module>":
        filename = frame.f_code.co_filename
        if filename.startswith(str(repo) + os.sep):
            relative = Path(filename).relative_to(repo).as_posix()
            if relative in expected_paths:
                stats["public_calls"] += 1
                public_paths.add(relative)
tree = ast.parse(source)
collect_public_bindings(tree)
tree = ast.fix_missing_locations(Instrument().visit(tree))
sys.setprofile(trace)
threading.setprofile(trace)
try:
    exec(compile(tree, str(Path(sys.argv[1]).resolve()), "exec"),
         {"__name__": "__main__", "_cooper_checked": checked, "_cooper_guarded": guarded,
          "_cooper_public_access": public_access})
finally:
    sys.setprofile(None)
    threading.setprofile(None)
    stats["public_paths"] = sorted(public_paths)
    stats["public_reads"] = sorted(public_reads)
    print("COOPER_PROBE_STATS=" + json.dumps(stats), flush=True)
'''


def _child(source_root, code, scratch, data_dir, label, paths):
    repo = scratch / label
    # Large public binary/runtime assets are deliberately exported as links to
    # the evaluator-owned read-only asset mount.  Following those links here
    # materializes the entire asset set into the container's bounded tmpfs for
    # every probe (and made the LlamaIndex public tree exhaust a 256 MiB /tmp).
    # Preserve the already-validated links in the fresh process-isolation tree;
    # they continue to resolve through the same read-only container mount.
    shutil.copytree(source_root, repo, symlinks=True)
    script = scratch / (label + ".py")
    script.write_text(code, encoding="utf-8")
    env = dict(os.environ)
    python_roots = [repo, repo / "src"]
    # A public target can live in a monorepo package root such as
    # ``llama-index-core/llama_index/...`` rather than at repository root or
    # under a conventional top-level ``src``.  Derive that import root only
    # from the already-validated public paths: walk outward through the Python
    # package chain, then add the first parent outside the chain.  The model
    # cannot inject an arbitrary PYTHONPATH entry.
    for relative in paths:
        target = repo.joinpath(*Path(relative).parts)
        if target.suffix != ".py" or not target.is_file():
            continue
        import_root = target.parent
        # PEP 420 namespace packages legitimately omit ``__init__.py`` (the
        # LlamaIndex tree is ``llama-index-core/llama_index/...``).  Walk
        # through both ordinary and namespace package directories while their
        # names are importable identifiers, but retain conventional source
        # roots themselves on PYTHONPATH.
        while (
            import_root != repo
            and import_root.name not in {"src", "lib", "python"}
            and import_root.name.isidentifier()
        ):
            import_root = import_root.parent
        if import_root == repo or repo in import_root.parents:
            python_roots.append(import_root)
    deduplicated_roots = []
    for root in python_roots:
        if root not in deduplicated_roots:
            deduplicated_roots.append(root)
    # The evaluator deliberately mounts the container-global /tmp as noexec.
    # Some public contracts exercise APIs that launch an executable created by
    # tempfile (for example, a custom editor script).  Give only this isolated
    # probe process an executable scratch directory under its existing
    # /workspace tree; keep the global /tmp security mount unchanged.
    probe_tmp = data_dir / "tmp"
    probe_tmp.mkdir(exist_ok=True)
    env.update(PYTHONPATH=os.pathsep.join(str(root) for root in deduplicated_roots),
               PYTHONDONTWRITEBYTECODE="1", PROBE_DATA_DIR=str(data_dir),
               TMPDIR=str(probe_tmp), TMP=str(probe_tmp), TEMP=str(probe_tmp))
    try:
        run = subprocess.run([sys.executable, "-c", CHILD, str(script), json.dumps(paths)], cwd=repo,
                             env=env, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return {"status": "unavailable", "error": "probe_process_timeout"}
    stats = {}
    marker = "COOPER_PROBE_STATS="
    marker_at = run.stdout.rfind(marker)
    if marker_at >= 0:
        encoded = run.stdout[marker_at + len(marker):].splitlines()[0]
        stats = json.loads(encoded)
    return {"status": "pass" if run.returncode == 0 else "fail",
            "returncode": run.returncode, **stats,
            "output": (run.stdout + "\n" + run.stderr)[-1800:]}


def run_plan(root, probes):
    results = []
    for probe in probes:
        row = {"probe_id": probe["probe_id"], "requirement_ids": probe["requirement_ids"],
               "paths": probe["paths"], "baseline": None, "candidate": None}
        sides = ["baseline", "candidate"] if probe["compare_baseline"] else ["candidate"]
        for side in sides:
            with tempfile.TemporaryDirectory(prefix="cooper-check-", dir=root) as temp:
                scratch = Path(temp)
                data_dir = scratch / "data"
                data_dir.mkdir()
                if probe["baseline_setup"]:
                    setup = _child(root / "baseline", probe["baseline_setup"],
                                   scratch, data_dir, "seed", probe["paths"])
                    if setup["status"] != "pass":
                        row[side] = {"status": "unavailable", "error": "baseline_setup_failed",
                                     "output": setup.get("output", setup.get("error", ""))}
                        continue
                result = _child(root / side, probe["code"], scratch, data_dir, "check", probe["paths"])
                kind = probe.get("kind", "runtime")
                observed = (set(result.get("public_reads", [])).intersection(probe["paths"])
                            if kind == "source" else (
                                result.get("public_calls", 0) > 0
                                or result.get("public_accesses", 0) > 0
                            ) and set(result.get("public_paths", [])).intersection(probe["paths"]))
                if result["status"] == "pass" and (
                    result.get("assertions", 0) + result.get("assertion_guards", 0) < 1 or not observed
                ):
                    result.update(status="unavailable", error="probe_executed_no_assertions_or_public_calls")
                row[side] = result
        results.append(row)
    return results


if __name__ == "__main__":
    root = Path.cwd()
    payload = json.loads((root / "probe_input.json").read_text(encoding="utf-8"))
    results = run_plan(root, payload["probes"])
    # One short final line fits the executor's bounded stdout tail. Detailed
    # failing traces remain here; successful output has no diagnostic value.
    for row in results:
        for side in ("baseline", "candidate"):
            if row[side] and row[side]["status"] == "pass":
                row[side].pop("output", None)
    encoded = json.dumps(results).encode("utf-8")
    (root / "probe_results.json").write_bytes(encoded)
    print("COOPER_BEHAVIOR_RESULTS_SHA256=" + hashlib.sha256(encoded).hexdigest(), flush=True)
