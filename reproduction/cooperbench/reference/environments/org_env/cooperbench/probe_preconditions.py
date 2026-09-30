"""Probe-owned observer evidence, without executing probe or candidate code."""
import ast
import re

OBSERVER_GUIDANCE = (
    "Audit the whole observer before authorizing product repair. A seeded RNG is a sequence, "
    "not a constant: do not call a state-advancing salt helper AFTER the operation to compute "
    "its earlier expected value. Capture same-call values or use an independently initialized "
    "reference RNG including setup calls. Prefix and suffix additions commute; reversing them "
    "does not duplicate the suffix. Toy tokenizer fixtures must satisfy existing BPE ranks and "
    "patterns before testing a new transform. Fake LM responses must implement the public "
    "adapter's structured fields/markers, not bare output labels. Evaluate every output assertion "
    "on one contract-conforming example: lower() membership and original case-sensitive count() "
    "can contradict. AST traversal must use nodes.Output, not type(...).__class__ (the metaclass). "
    "from_string TemplateModule does not inherently own a loader/get_source API. Initialize "
    "gettext using the public translation fixture API; old-style gettext returns untranslated "
    "format text for later substitution, not x%kwargs when kwargs is empty. Trimmed translation "
    "does not collapse repeated interior spaces. Plain template literals use {% ... %}, not "
    "{%% ... %%}, unless subsequent Python percent formatting actually consumes the escapes. "
    "Ordinary JPEG is lossy: observe pre-encoding marking or justify codec-aware observations, "
    "still checking all required formats; do not force unrelated encoder quality changes. "
    "Lock tests need explicit readiness/release synchronization, bounded deadlines and finally "
    "cleanup. Startup sleep guesses are not synchronization. Observe while the editor is known "
    "to run, not after its shorter sleep ends. Captured observer_evidence is not candidate "
    "approval. Explain each exact precondition causally; generic feature quotes cannot override "
    "a demonstrated contradiction. Correct probes through the actual peer workflow; do not "
    "weaken product requirements, catch unexpected failures, or edit product code to fit a bad fixture. "
    "Keep input object types distinct from access protocols: a dict with key 'name' does not "
    "have attribute .name. Grouping that preserves the input entries cannot make hasattr(dict, "
    "'name') true. An editor script must contain actual newline bytes after its shebang and "
    "commands; a decoded literal backslash-n is not a newline. Account for all public constructor "
    "setup calls before testing a fake LM's operation-call counter. Plural translation fixtures "
    "must initialize ngettext as well as gettext through the supported public configuration. "
    "A settings/configuration registry does not automatically reconfigure an existing cache "
    "object. Trace the public cache configuration owner (for example configure_cache or the "
    "cache constructor) before asserting which memory/disk path an operation should take. "
)

def _template_lexical_error(value):
    position = 0
    while True:
        start = value.find('{%', position)
        if start < 0:
            break
        if value[start + 2:start + 3] == '%':
            return 'literal_double_percent_tag_without_percent_format'
        quote, escaped = '', False
        cursor = start + 2
        while cursor < len(value):
            char = value[cursor]
            if escaped:
                escaped = False
            elif quote and char == '\\':
                escaped = True
            elif quote:
                if char == quote:
                    quote = ''
            elif char in {'"', "'"}:
                quote = char
            elif value[cursor:cursor + 2] == '%}':
                position = cursor + 2
                break
            cursor += 1
        else:
            if quote:
                return 'decoded_template_tag_has_unterminated_string'
            break
        if quote:
            return 'decoded_template_tag_has_unterminated_string'
    return ''

def observer_evidence(probe, context, site):
    code = str(probe.get('code') or '')
    try:
        tree = ast.parse(code[:80000])
    except SyntaxError:
        return []
    nodes, rows = list(ast.walk(tree))[:2000], []
    lines = set(site.get('probe_lines') or [])
    syntax = str(site.get('exception_type') or '').endswith('SyntaxError')
    assignments = {}
    for n in nodes:
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            assignments.setdefault(n.id, []).append(n)
    constants = {n.targets[0].id: (n.value.value, n.lineno) for n in tree.body
                 if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and isinstance(n.targets[0], ast.Name) and len(assignments[n.targets[0].id]) == 1
                 and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}
    def add(node, kind, consequence, contradiction=False):
        row = dict(line=node.lineno, kind=kind, probe_quote=(ast.get_source_segment(code, node) or '')[:1200],
                   consequence=consequence, mechanical_contradiction=contradiction)
        if row not in rows and len(rows) < 16:
            rows.append(row)
            return row
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
        if name in {'parse', 'from_string', 'lex'}:
            for arg in node.args:
                value = arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None
                if isinstance(arg, ast.Name) and arg.id in constants and constants[arg.id][1] < node.lineno:
                    value = constants[arg.id][0]
                error = _template_lexical_error(value) if isinstance(value, str) else ''
                if error:
                    row = add(node, error, 'Decoded template has a lexical defect before the intended assertion.', syntax and node.lineno in lines)
                    if row is not None:
                        row['decoded_value'] = value[:2048]
        if name == 'find_all' and any(isinstance(a, ast.Attribute) and a.attr == '__class__' and isinstance(a.value, ast.Call) and isinstance(a.value.func, ast.Name) and a.value.func.id == 'type' for a in node.args):
            add(node, 'metaclass_not_requested_ast_node', 'The argument is type, not the public AST node class.')
        if name == 'hasattr' and len(node.args) > 1 and isinstance(node.args[1], ast.Constant) and any(isinstance(n, ast.Dict) for n in nodes) and 'groupby' in code:
            add(node, 'mapping_key_is_not_object_attribute', 'Trace the preserved input entry type: dictionary keys are not object attributes.')
        if name == 'write' and node.args:
            argument = node.args[0]
            value = argument.value if isinstance(argument, ast.Constant) else None
            if isinstance(argument, ast.JoinedStr) and argument.values and isinstance(argument.values[0], ast.Constant):
                value = argument.values[0].value
            if isinstance(value, str) and value.startswith('#!') and '\\n' in value.split('\n', 1)[0]:
                row = add(node, 'script_shebang_contains_literal_backslash_n', 'The decoded file header has backslash-n characters, not a shebang line separator.')
                if row is not None:
                    row['decoded_value'] = value[:2048]
                    row['decoded_value_complete'] = isinstance(argument, ast.Constant)
        if name == 'GroundedProposer' and 'call_count' in code and any(k.arg == 'trainset' for k in node.keywords):
            add(node, 'fake_counter_includes_constructor_setup', 'Constructor dataset-description and adapter retries consume fake outputs before proposal/rephrase assertions.')
        if name == 'configure' and any(k.arg in {'enable_memory_cache','enable_disk_cache','cache_dir','disk_cache_dir'} for k in node.keywords):
            add(node, 'settings_registry_is_not_cache_configuration', 'Establish from public source that these settings actually configure the existing cache object.')
        if name in {'from_string','render'} and 'pluralize' in code and 'gettext' in code and not any(s in code for s in ('ngettext','install_gettext','install_null_translations')):
            add(node, 'plural_translation_fixture_missing_ngettext', 'The plural branch needs the supported ngettext fixture, not only singular gettext.')
        if name == '_cache_salt':
            add(node, 'stateful_expectation_call', 'A second RNG-consuming helper call observes a different sequence element.')
        if name == 'sleep' and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, (int, float)) and 0 < node.args[0].value <= .2:
            add(node, 'short_sleep_is_not_process_readiness', 'No readiness/release handshake is established by this sleep.')
        if name == 'lower' and '.count(' in code:
            add(node, 'case_sensitive_observer', 'Evaluate lowercased membership and case-sensitive counts together on one conforming output.')
        if name == 'get_source' and '__loader__' in (ast.get_source_segment(code, node.func) or ''):
            add(node, 'assumed_template_module_loader', 'Establish the actual object supports this public loader protocol.')
        if name == 'render' and 'gettext' in code and not any(s in code for s in ('install_gettext', 'install_null_translations', 'env.globals', 'environment.globals')):
            add(node, 'translation_fixture_initialization', 'Rendering needs the supported public gettext fixture.')
        if name == 'getpixel' and re.search(r'\bJPEG\b', code):
            add(node, 'lossy_codec_exact_pixel_observer', 'Pre-encoding red marks need not survive JPEG round trips byte-for-byte.')
    return rows

def grounded_observer_basis(material, value):
    evidence = material.get('observer_evidence') or []
    if not evidence:
        return []
    if not isinstance(value, list) or len(value) != len(evidence):
        return None
    out = []
    for item in evidence:
        matches = [r for r in value if isinstance(r, dict) and all(r.get(k) == item.get(k) for k in ('line', 'kind', 'probe_quote'))]
        if len(matches) != 1 or not isinstance(matches[0].get('reason'), str) or not matches[0]['reason'].strip() or len(matches[0]['reason']) > 1600:
            return None
        if item.get('mechanical_contradiction'):
            return None
        out.append({k: matches[0][k] for k in ('line', 'kind', 'probe_quote', 'reason')})
    return out
