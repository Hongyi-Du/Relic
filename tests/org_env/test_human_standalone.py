"""The workspace runs on its own (HCI standalone).

The claim this rests on is that the HCI server needs almost nothing: its import
graph is the standard library plus this repository, and serving it takes only
FastAPI and uvicorn. The installed image also carries the OpenAI-compatible
SDK needed by the opt-in real-model path, without the experiment stack.

It is also easy to break by accident -- one convenience import of pandas
somewhere in org_env and the light install stops working, silently, until
someone tries it on a clean machine.

Run:  PYTHONPATH="." python tests/org_env/test_human_standalone.py
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

#: Heavy packages in pyproject.toml that the workspace must never pull in.
FORBIDDEN = ("litellm", "faiss", "datasets", "pandas", "numpy", "scipy",
             "matplotlib", "seaborn", "torch", "gradio", "django", "sklearn",
             "networkx", "together")


def test_the_server_imports_without_any_third_party_package():
    """A subprocess, because import state leaks between tests in one process."""
    probe = (
        "import sys;"
        "before = set(m.split('.')[0] for m in sys.modules);"
        "import environments.org_env.backend.main;"
        "new = set(m.split('.')[0] for m in sys.modules) - before;"
        "third = sorted(m for m in new if not m.startswith('_') "
        "  and m not in sys.stdlib_module_names "
        "  and m not in ('agent_sdk', 'environments', 'relic'));"
        "print(','.join(third))"
    )
    out = subprocess.run([sys.executable, "-c", probe], cwd=REPO, capture_output=True,
                         text=True, env={**_env(), "ORG_LLM": "0"}, timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    third = [x for x in out.stdout.strip().split(",") if x]
    assert third == [], f"the workspace now imports third-party packages: {third}"


def test_serving_it_needs_only_fastapi_and_uvicorn():
    probe = (
        "import sys;"
        "from environments.org_env.backend import main;"
        "before = set(m.split('.')[0] for m in sys.modules);"
        "main.build_app(); main.api_sim_run_ticks(2); main.HUMAN.members();"
        "new = sorted(m for m in (set(m.split('.')[0] for m in sys.modules) - before) "
        "  if not m.startswith('_') and m not in sys.stdlib_module_names);"
        "print(','.join(new)); main.HUMAN.shutdown()"
    )
    out = subprocess.run([sys.executable, "-c", probe], cwd=REPO, capture_output=True,
                         text=True, env={**_env(), "ORG_LLM": "0"}, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    pulled = {x for x in out.stdout.strip().split(",") if x}

    heavy = sorted(p for p in FORBIDDEN if p in pulled)
    assert not heavy, f"serving the workspace now needs {heavy}"
    assert "fastapi" in pulled and "starlette" in pulled, pulled


def test_the_light_requirements_file_covers_what_is_actually_needed():
    text = (REPO / "requirements-hci.txt").read_text(encoding="utf-8")
    required = [ln.split(">=")[0].split("[")[0].strip()
                for ln in text.splitlines()
                if ln.strip() and not ln.startswith("#")]

    assert "fastapi" in required and "uvicorn" in required, required
    # Anything heavy in here would defeat the point of a separate file.
    assert not [p for p in FORBIDDEN if p in required], required
    # The SDK is installed even though the default run needs no API key.
    assert "openai" in required


def test_the_entry_point_parses_its_arguments():
    from tools.run_hci import parse_args

    default = parse_args([])
    assert default.host == "127.0.0.1", "must not bind wide without being asked"
    assert default.llm is False, "must not need an API key by default"
    assert default.warmup > 0 and default.tick_seconds > 0

    wide = parse_args(["--host", "0.0.0.0", "--port", "9000", "--llm", "--paused"])
    assert (wide.host, wide.port, wide.llm, wide.paused) == ("0.0.0.0", 9000, True, True)


def test_the_entry_point_starts_and_serves(tmp_path=None):
    """Actually run it and fetch the page, so a broken import or a bad flag
    fails here rather than on someone's machine."""
    import socket
    import time
    import urllib.request

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    proc = subprocess.Popen(
        [sys.executable, "tools/run_hci.py", "--port", str(port),
         "--warmup", "2", "--tick-seconds", "30", "--paused"],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**_env(), "ORG_LLM": "0"})
    try:
        deadline = time.time() + 120
        body = ""
        while time.time() < deadline:
                if proc.poll() is not None:
                    raise AssertionError(f"exited early:\n{proc.stdout.read()[-2000:]}")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/", timeout=2) as r:
                        body = r.read().decode()
                        final_url = r.geturl()
                    break
                except Exception:
                    time.sleep(0.5)
        assert body, "the workspace never came up"
        assert "LanternForge" in body
        assert final_url.endswith("/org/setup")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_the_docker_files_describe_the_same_light_install():
    dockerfile = (REPO / "Dockerfile.hci").read_text(encoding="utf-8")
    assert "requirements-hci.txt" in dockerfile
    assert "pyproject.toml" not in dockerfile, "that would install the whole stack"
    assert "tools/run_hci.py" in dockerfile

    # BuildKit looks for "<dockerfile>.dockerignore"; a plain ".dockerignore.hci"
    # is silently ignored and the frozen corpora end up in the build context.
    ignore = (REPO / "Dockerfile.hci.dockerignore").read_text(encoding="utf-8")
    # The frozen OSS corpora are hundreds of megabytes and never read here.
    assert "environments/org_env/data/" in ignore
    assert "benchmarks/" in ignore

    compose = (REPO / "docker-compose.hci.yml").read_text(encoding="utf-8")
    assert "Dockerfile.hci" in compose
    assert "127.0.0.1:8100:8100" in compose, "must not publish wide by default"


def test_the_windows_script_covers_the_no_python_case():
    """Someone on a fresh Windows box is the whole point of this script; it has
    to create its own environment and say something useful when it cannot."""
    script = (REPO / "tools" / "run_hci.ps1").read_text(encoding="utf-8")

    assert ".venv-hci" in script, "must be able to build its own environment"
    assert "requirements-hci.txt" in script
    assert "python.org/downloads/windows" in script, "must say where to get Python"
    assert "docker compose" in script, "must offer the Docker way out"
    # Probing for an interpreter fails by design; under ErrorActionPreference
    # Stop a native command writing to stderr would abort the whole script.
    assert "Invoke-Probe" in script
    assert "$ErrorActionPreference = 'Continue'" in script


def test_nothing_in_the_standalone_path_mentions_wsl():
    """The point of this path is that Windows users never open a WSL shell."""
    for name in ("tools/run_hci.py", "tools/run_hci.ps1", "Dockerfile.hci",
                 "docker-compose.hci.yml", "requirements-hci.txt"):
        text = (REPO / name).read_text(encoding="utf-8").lower()
        for line in text.splitlines():
            if "wsl" not in line:
                continue
            # Saying "no WSL needed" is fine; requiring one is not.
            assert any(w in line for w in ("no wsl", "without", "never", "needs no")), (
                f"{name} appears to require WSL: {line.strip()}")


def _env():
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env.pop("ORG_PRODUCT_SUBSTRATE", None)
    env.pop("ORG_OSS_MODE", None)
    return env


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("  ok ", fn.__name__)
    print(f"All {len(fns)} standalone tests passed!")
