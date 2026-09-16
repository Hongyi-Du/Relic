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
EXPECTED = [{'module': 'blobstore.objects', 'functions': {}, 'classes': {'InvalidObjectId': {}, 'CorruptObjectError': {}, 'ObjectStore': {'__init__': {'parameters': ['self', 'root'], 'property': False}, 'validate_id': {'parameters': ['object_id'], 'property': False}, 'path_for': {'parameters': ['self', 'object_id'], 'property': False}, 'put_bytes': {'parameters': ['self', 'data'], 'property': False}, 'put_stream': {'parameters': ['self', 'source'], 'property': False}, 'get': {'parameters': ['self', 'object_id'], 'property': False}, 'exists': {'parameters': ['self', 'object_id'], 'property': False}, 'verify': {'parameters': ['self', 'object_id'], 'property': False}, 'iter_ids': {'parameters': ['self'], 'property': False}}}, 'constants': []}]
PUBLIC_CASES = [('digest_and_roundtrip', "\nimport hashlib\nfrom blobstore.objects import ObjectStore\ns=ObjectStore(ROOT); data=b'hello\\x00world'; oid=s.put_bytes(data)\nassert oid==hashlib.sha256(data).hexdigest() and s.get(oid)==data and s.verify(oid)\n"), ('sharded_path', "\nfrom blobstore.objects import ObjectStore\ns=ObjectStore(ROOT); oid=s.put_bytes(b'x'); p=s.path_for(oid)\nassert p.parent.name==oid[:2] and p.name==oid[2:] and p.is_file()\n")]


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

print("PASS step 1 API surface and public behavior cases")
