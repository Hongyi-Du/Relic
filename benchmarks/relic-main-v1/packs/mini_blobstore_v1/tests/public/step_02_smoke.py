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
EXPECTED = [{'module': 'blobstore.manifests', 'functions': {}, 'classes': {'ManifestConflictError': {}, 'Manifest': {'__init__': {'parameters': ['self', 'name', 'objects', 'metadata'], 'property': False}}, 'ManifestStore': {'__init__': {'parameters': ['self', 'root', 'objects'], 'property': False}, 'validate_name': {'parameters': ['name'], 'property': False}, 'path_for': {'parameters': ['self', 'name'], 'property': False}, 'create': {'parameters': ['self', 'name', 'object_ids', 'metadata'], 'property': False}, 'get': {'parameters': ['self', 'name'], 'property': False}, 'delete': {'parameters': ['self', 'name'], 'property': False}, 'iter_names': {'parameters': ['self'], 'property': False}, 'referenced_ids': {'parameters': ['self'], 'property': False}}}, 'constants': []}]
PUBLIC_CASES = [('create_get_canonical', "\nimport json\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\no=ObjectStore(ROOT); oid=o.put_bytes(b'x'); m=ManifestStore(ROOT,o)\nr=m.create('team/data',[oid],{'z':1,'a':2}); assert r.objects==(oid,)\np=m.path_for('team/data'); assert p.read_bytes()==(json.dumps(json.loads(p.read_text()),sort_keys=True,separators=(',',':'))+'\\n').encode()\n"), ('unsafe_names_rejected', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nm=ManifestStore(ROOT,ObjectStore(ROOT))\nfor name in ['', '/abs', '../x', 'a/../b', 'a//b']:\n    try: m.create(name,[])\n    except ValueError: pass\n    else: raise AssertionError(name)\nassert not list((ROOT/'manifests').rglob('*.json'))\n")]


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

print("PASS step 2 API surface and public behavior cases")
