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
EXPECTED = [{'module': 'blobstore.replicas', 'functions': {}, 'classes': {'ReplicaStatus': {'__init__': {'parameters': ['self', 'object_id', 'states'], 'property': False}}, 'ReplicaSet': {'__init__': {'parameters': ['self', 'primary', 'replicas'], 'property': False}, 'audit': {'parameters': ['self', 'object_id'], 'property': False}, 'repair': {'parameters': ['self', 'object_id', 'replica_index'], 'property': False}, 'replicate_all': {'parameters': ['self'], 'property': False}}}, 'constants': []}]
PUBLIC_CASES = [('audit_states', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.replicas import ReplicaSet\np=ObjectStore(ROOT/'p'); a=ObjectStore(ROOT/'a'); b=ObjectStore(ROOT/'b'); oid=p.put_bytes(b'x'); a.put_bytes(b'x'); b.put_bytes(b'x'); b.path_for(oid).write_bytes(b'bad')\nr=ReplicaSet(p,[a,b]).audit(oid); assert r.states==('healthy','healthy','corrupt')\n"), ('replicate_missing', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.replicas import ReplicaSet\np=ObjectStore(ROOT/'p'); r=ObjectStore(ROOT/'r'); ids=[p.put_bytes(x) for x in (b'a',b'b')]\nassert ReplicaSet(p,[r]).replicate_all()==2 and all(r.verify(x) for x in ids)\n")]


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

print("PASS step 4 API surface and public behavior cases")
