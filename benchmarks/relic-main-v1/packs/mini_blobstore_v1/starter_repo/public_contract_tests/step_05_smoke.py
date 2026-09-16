#!/usr/bin/env python3
"""Generated public callable-surface and representative behavior smoke."""

from __future__ import annotations

import importlib
import inspect
import pathlib
import sys
import tempfile


WORKSPACE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
EXPECTED = [{'module': 'blobstore.gc', 'functions': {}, 'classes': {'GCPlan': {'__init__': {'parameters': ['self', 'candidates', 'state_fingerprint'], 'property': False}}, 'GCAudit': {'__init__': {'parameters': ['self', 'reachable', 'orphaned', 'corrupt'], 'property': False}}, 'GarbageCollector': {'__init__': {'parameters': ['self', 'root', 'objects', 'manifests', 'uploads'], 'property': False}, 'reachable': {'parameters': ['self'], 'property': False}, 'plan': {'parameters': ['self'], 'property': False}, 'apply': {'parameters': ['self', 'plan'], 'property': False}, 'audit': {'parameters': ['self'], 'property': False}}}, 'constants': []}]
PUBLIC_CASES = [('reachability_manifest_and_upload', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore\nfrom blobstore.gc import GarbageCollector\no=ObjectStore(ROOT); m=ManifestStore(ROOT,o); u=UploadStore(ROOT,o,m)\na=o.put_bytes(b'a'); b=o.put_bytes(b'b'); c=o.put_bytes(b'c'); m.create('keep',[a]); uid=u.begin('pending'); u.upload_part(uid,1,b'b')\ngc=GarbageCollector(ROOT,o,m,u); assert gc.reachable()=={a,b} and gc.plan().candidates==(c,)\n"), ('plan_is_sorted_read_only', "\nimport hashlib\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore\nfrom blobstore.gc import GarbageCollector\no=ObjectStore(ROOT); [o.put_bytes(x) for x in (b'z',b'a',b'm')]; gc=GarbageCollector(ROOT,o,ManifestStore(ROOT,o),UploadStore(ROOT,o,ManifestStore(ROOT,o)))\nbefore={x:hashlib.sha256(o.path_for(x).read_bytes()).hexdigest() for x in o.iter_ids()}; p=gc.plan()\nassert p.candidates==tuple(sorted(p.candidates)) and before=={x:hashlib.sha256(o.path_for(x).read_bytes()).hexdigest() for x in o.iter_ids()}\n")]


def parameter_names(callable_object):
    return list(inspect.signature(callable_object).parameters)


for module_contract in EXPECTED:
    module = importlib.import_module(module_contract["module"])
    for name, parameters in module_contract["functions"].items():
        function = getattr(module, name)
        assert parameter_names(function) == parameters, (name, parameter_names(function), parameters)
    for class_name, methods in module_contract["classes"].items():
        cls = getattr(module, class_name)
        for method_name, method_contract in methods.items():
            raw_method = inspect.getattr_static(cls, method_name)
            if method_contract["property"]:
                assert isinstance(raw_method, property), (
                    f"{class_name}.{method_name}",
                    type(raw_method).__name__,
                    "property",
                )
                continue
            if isinstance(raw_method, (classmethod, staticmethod)):
                method = raw_method.__func__
            else:
                method = raw_method
            parameters = method_contract["parameters"]
            assert parameter_names(method) == parameters, (
                f"{class_name}.{method_name}",
                parameter_names(method),
                parameters,
            )
    for name in module_contract["constants"]:
        assert hasattr(module, name), name


for case_name, case_body in PUBLIC_CASES:
    with tempfile.TemporaryDirectory(prefix="public-contract-") as raw:
        ROOT = pathlib.Path(raw)
        exec(case_body, {"ROOT": ROOT, "__builtins__": __builtins__})
    print(f"PASS public behavior case: {case_name}")

print("PASS step 5 API surface and public behavior cases")
