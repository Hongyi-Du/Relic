"""Private compatibility namespace for source-derived Relic modules.

The public project is Relic.  A small number of upstream OrgEnv modules use
``society_core.<module>`` imports, so this namespace intentionally exposes
only the source files needed by the shipped release.  It must stay a thin
import boundary rather than a second runtime or a public API.
"""
