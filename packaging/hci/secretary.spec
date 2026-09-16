from pathlib import Path
import os

from PyInstaller.utils.hooks import collect_submodules


repo = Path(SPECPATH).parents[1]
secret_config = Path(os.environ["SECRETARY_LLM_CONFIG"])
pack_root = repo / "benchmarks" / "relic-main-v1" / "packs" / "traffic_watch_v1"
pack_destination = "benchmarks/relic-main-v1/packs/traffic_watch_v1"


def source_tree(source: Path, destination: str):
    return [
        (
            str(path),
            str(Path(destination) / path.relative_to(source).parent).replace("\\", "/"),
        )
        for path in source.rglob("*")
        if path.is_file()
        and not {"__pycache__", ".pytest_cache"}.intersection(path.parts)
        and path.suffix != ".pyc"
    ]

datas = [
    (str(repo / "config" / "llm.yaml"), "config"),
    (str(secret_config), "config"),
    (
        str(repo / "environments" / "org_env" / "frontend" / "app" / "dist"),
        "environments/org_env/frontend/app/dist",
    ),
    (str(pack_root / "manifest.yaml"), pack_destination),
    (str(pack_root / "issues" / "public"), f"{pack_destination}/issues/public"),
    (str(repo / "agent_sdk" / "prompts"), "agent_sdk/prompts"),
] + source_tree(pack_root / "starter_repo", f"{pack_destination}/starter_repo")

hiddenimports = [
    "environments.org_env.llm.config",
    "environments.org_env.product.substrates.loader",
    "environments.org_env.product.substrates.oss_time_machine",
    "uvicorn.logging",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.lifespan.on",
    "cv2",
    "numpy",
    "yaml",
] + collect_submodules("flask") + collect_submodules("openai") + collect_submodules("pytest")

a = Analysis(
    [str(repo / "packaging" / "hci" / "secretary_launcher.py")],
    pathex=[str(repo)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "pandas",
        "scipy",
        "sklearn",
        "torch",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Secretary",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Secretary",
)
