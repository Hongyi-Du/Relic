"""Acceptance check for issue_file_discovery_depth.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: iter_find_files accepts a max_depth argument that caps how many directory levels below the starting directory are searched (0 meaning the starting directory only); omitting it keeps the current full-tree behaviour and the cap composes with the existing pattern and ignore arguments.
"""
from unittest.mock import patch

from boltons import fileutils


def test_iter_find_files_max_depth_limits_walk_and_composes_with_filters():
    tree = {
        "/project": (["child"], ["root.py", "ignored.py", "root.txt"]),
        "/project/child": (["deep"], ["child.py", "ignored.py", "child.txt"]),
        "/project/child/deep": ([], ["deep.py", "ignored.py", "deep.txt"]),
    }

    def fake_walk(directory, *args, **kwargs):
        pending = [directory]
        while pending:
            root = pending.pop(0)
            dirnames, filenames = tree[root]
            dirnames = list(dirnames)
            yield root, dirnames, list(filenames)
            pending.extend(
                "/".join((root, dirname)) for dirname in dirnames
            )

    with patch.object(fileutils.os, "walk", fake_walk):
        assert list(fileutils.iter_find_files(
            "/project", "*.py", ignored="ignored.py", max_depth=0
        )) == ["/project/root.py"]

        assert list(fileutils.iter_find_files(
            "/project", "*.py", ignored="ignored.py", max_depth=1
        )) == ["/project/root.py", "/project/child/child.py"]

        assert list(fileutils.iter_find_files(
            "/project", "*.py", ignored="ignored.py"
        )) == [
            "/project/root.py",
            "/project/child/child.py",
            "/project/child/deep/deep.py",
        ]
