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
EXPECTED = [{'module': 'blobstore.uploads', 'functions': {}, 'classes': {'UploadConflictError': {}, 'UploadStore': {'__init__': {'parameters': ['self', 'root', 'objects', 'manifests'], 'property': False}, 'begin': {'parameters': ['self', 'target', 'metadata'], 'property': False}, 'upload_part': {'parameters': ['self', 'upload_id', 'number', 'data'], 'property': False}, 'complete': {'parameters': ['self', 'upload_id', 'part_count'], 'property': False}, 'incomplete_object_ids': {'parameters': ['self'], 'property': False}}}, 'constants': []}]
PUBLIC_CASES = [('multipart_complete_order', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore\no=ObjectStore(ROOT); m=ManifestStore(ROOT,o); u=UploadStore(ROOT,o,m); uid=u.begin('release/v1',{'k':'v'})\nu.upload_part(uid,2,b'world'); u.upload_part(uid,1,b'hello ')\nr=u.complete(uid,2); assert o.get(r['object_id'])==b'hello world' and m.get('release/v1').metadata=={'k':'v'}\n"), ('part_retry_and_conflict', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore,UploadConflictError\no=ObjectStore(ROOT); u=UploadStore(ROOT,o,ManifestStore(ROOT,o)); uid=u.begin('x')\na=u.upload_part(uid,1,b'a'); assert u.upload_part(uid,1,b'a')==a\ntry: u.upload_part(uid,1,b'b')\nexcept UploadConflictError: pass\nelse: raise AssertionError('changed part accepted')\n")]


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

print("PASS step 3 API surface and public behavior cases")
