"""A public-source index, never a replacement specification or hidden oracle."""
from __future__ import annotations

import re


def _public_clauses(text: str) -> list[str]:
    """Split prose modality boundaries, without splitting inline Python."""
    text = str(text or "")
    masked = re.sub(r"`[^`]*`", lambda match: "\u00a7" * len(match.group()), text)
    boundary = re.compile(
        r"[;\u3002\uff1b\uff01\uff1f]|(?<=[.!?])\s+|"
        r",\s*(?=(?:optionally|may\b|might\b|but\b|however\b))|"
        r"\s+\b(?:but|however)\b\s+|"
        r"\s+and\s+(?=(?:optionally|may\b|might\b|must\b|shall\b))",
        re.IGNORECASE,
    )
    clauses, start = [], 0
    for match in boundary.finditer(masked):
        piece = text[start:match.start()].strip(" ,")
        if piece:
            clauses.append(piece)
        start = match.end()
    tail = text[start:].strip(" ,")
    if tail:
        clauses.append(tail)
    return clauses


def _public_sentences(text: str) -> list[str]:
    """Split prose sentences while preserving coupled semicolon clauses.

    Semicolon neighbors often define one compatibility invariant (for example
    an unchanged memory layer paired with disk-only behavior).  Splitting that
    pair makes the compatibility ledger select only half of the promise.  The
    false coverage observed in Cooper came from multiple full sentences being
    credited together, so sentence boundaries are the safe granularity here.
    """

    text = str(text or "")
    masked = re.sub(r"`[^`]*`", lambda match: "\u00a7" * len(match.group()), text)
    boundary = re.compile(r"(?<=[.!?])\s+|[\u3002\uff01\uff1f]")
    sentences: list[str] = []
    start = 0
    for match in boundary.finditer(masked):
        piece = text[start:match.start()].strip(" ,")
        if piece:
            sentences.append(piece)
        start = match.end()
    tail = text[start:].strip(" ,")
    if tail:
        sentences.append(tail)
    return sentences


def _discretionary_clause(clause: str) -> bool:
    prose = re.sub(r"`[^`]*`", " ", clause)
    # Optional parameters describe accepted values, not permission to omit the
    # entrypoint. Only explicit discretionary *modality* exempts an API clause.
    discretionary = re.search(
        r"\b(?:optionally|may(?!\s+not\b)|might|need\s+not|"
        r"not\s+(?:required|mandatory)|(?:is|are|remains?)\s+optional)\b|"
        r"^\s*(?:\*\*)?optional\b", prose, re.IGNORECASE,
    )
    if discretionary is None:
        return False
    # An optional feature still has real requirements when enabled, and a
    # prohibition ('may not', 'must not') is not discretionary permission.
    binding_prose = re.sub(r"\bnot\s+(?:required|mandatory)\b", "", prose, flags=re.IGNORECASE)
    binding = re.search(
        r"\b(?:must|shall|required|mandatory|never|forbidden|prohibited|may\s+not)\b|"
        r"\u5fc5\u987b|\u4e0d\u5f97|\u7981\u6b62", binding_prose, re.IGNORECASE,
    )
    return binding is None


def required_public_clauses(text: str) -> list[str]:
    """Fallible mechanical scope for compulsory examples, not a new contract.

    Keep mandatory and conditional neighbors intact. The source index and full
    request remain authoritative, including all optional language we omit here.
    """
    return [clause for clause in _public_clauses(text) if not _discretionary_clause(clause)]


def purely_optional_requirement(text: str) -> bool:
    clauses = _public_clauses(text)
    return bool(clauses) and all(_discretionary_clause(clause) for clause in clauses)


def acceptance_index(description: str) -> list[dict]:
    """Index public prose with exact line provenance and Markdown context.

    Fenced examples stay in the authoritative request, rather than becoming
    disconnected docstring/parameter requirements. No text or row-count cap
    silently changes the public contract. Do not classify requirements by an
    English verb whitelist: a correction phrased as "recalculates" is just as
    binding as one phrased as "must return". Context sections remain in this
    source index, but are not compulsory-credit obligations below.
    """
    rows: list[dict] = []
    pending: list[str] = []
    start = end = 0
    section = ""
    fence = ""

    def flush():
        if not pending:
            return
        text = " ".join(pending)
        pending.clear()
        # Titles and lead-ins introduce the following requirements; they
        # cannot be proven independently by an executable check.
        if text.rstrip("* _").endswith(":"):
            return
        if section.casefold() not in {"files modified", "files changed"}:
            rows.append({"text": text, "start_line": start, "end_line": end,
                         "section": section})

    for number, raw in enumerate(str(description or "").splitlines(), 1):
        stripped = raw.strip()
        marker = re.match(r"^(`{3,}|~{3,})", stripped)
        if marker:
            flush()
            if not fence:
                fence = marker.group(1)
            elif marker.group(1)[0] == fence[0] and len(marker.group(1)) >= len(fence):
                fence = ""
            continue
        if fence:
            continue
        if not stripped or re.fullmatch(r"[-=_*]{3,}", stripped):
            flush()
            continue
        if re.match(r"^(?:\*\*)?Files? (?:Modified|Changed)\b", stripped, re.IGNORECASE):
            flush()
            section = "Files Modified"
            continue
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*$", stripped)
        if heading:
            flush()
            section = heading.group(1)
            continue
        bullet = re.match(r"^(?:[-+*]|\d+[.)])\s+(.+)$", stripped)
        if bullet:
            flush()
            stripped = bullet.group(1)
        # A standalone bold heading, including a colon outside its emphasis.
        if re.fullmatch(r"\*\*[^*]+\*\*:?", stripped):
            flush()
            section = stripped.strip("*: ")
            continue
        labelled = re.match(r"^\*\*([^*]+)\*\*\s*[:\u2013-]\s*(.*)$", stripped)
        if labelled and labelled.group(1).strip().casefold() in {
            "title", "background", "technical background", "problem", "context",
            "motivation", "overview", "description", "solution", "expected behavior",
        }:
            flush()
            if labelled.group(1).strip().casefold() == "title":
                continue
            section = labelled.group(1).strip()
            stripped = labelled.group(2).strip()
            if not stripped:
                continue
        # Public briefs also use unadorned section labels such as "Python
        # Compatibility". Keep labels out of compulsory executable coverage.
        if re.fullmatch(r"[\w -]+(?:Requirements|Compatibility)", stripped):
            flush()
            section = stripped
            continue
        if not pending:
            start = number
        end = number
        pending.append(stripped)
        if stripped.rstrip("* _").endswith(":"):
            flush()
        elif stripped.rstrip("* _").endswith((".", "!", "?")):
            flush()
    flush()
    return rows


def indexed_obligations(description: str) -> list[str]:
    """Return atomic, source-derived obligations for executable review.

    ``acceptance_index`` deliberately groups consecutive prose into readable
    source spans.  A single grouped row can nevertheless contain several
    independently observable requirements.  Crediting that whole row to one
    callback assertion previously hid a preceding ``trim`` transformation in a
    real Cooper task: the probe, peer review, and joint replay all passed while
    the official evaluator correctly rejected the untrimmed result.

    Split only at sentence boundaries. Coupled semicolon clauses remain intact
    so a compatibility promise cannot lose its scope. The complete request and
    coarse source spans remain available to both model stages, so this is a
    coverage index rather than a rewritten specification. Advisory clauses
    remain non-compulsory.
    """

    obligations: list[str] = []
    for row in acceptance_index(description):
        if row["section"].casefold() in {
            "background", "technical background", "problem", "context", "motivation",
            "current behavior",
        }:
            continue
        clauses = _public_sentences(row["text"])
        # Preserve the exact historical row (including terminal punctuation)
        # when it is already atomic. Only multi-clause rows need finer ids.
        if len(clauses) == 1:
            clauses = [row["text"]]
        for clause in clauses:
            if clause and not purely_optional_requirement(clause):
                obligations.append(clause)
    return list(dict.fromkeys(obligations))
