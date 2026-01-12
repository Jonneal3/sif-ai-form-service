from __future__ import annotations

import re
from typing import Iterable, List

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "the",
    "to",
    "with",
    "your",
}


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) >= 3]


def _bigrams(tokens: List[str]) -> List[str]:
    return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


def extract_service_anchor_terms(
    industry: str,
    service: str,
    grounding: str,
    max_terms: int = 30,
    min_terms: int = 15,
) -> List[str]:
    """
    Lightweight keyword extractor for service-relevant anchors.
    """
    sources = [industry or "", service or "", grounding or ""]
    tokens: List[str] = []
    phrases: List[str] = []
    for text in sources:
        toks = _tokenize(text)
        if not toks:
            continue
        tokens.extend(toks)
        phrases.extend(_bigrams(toks))

    candidates = _dedupe_keep_order(phrases + tokens)
    if max_terms > 0:
        candidates = candidates[:max_terms]

    if min_terms and len(candidates) < min_terms:
        for tok in _dedupe_keep_order(tokens):
            if tok in candidates:
                continue
            candidates.append(tok)
            if max_terms > 0 and len(candidates) >= max_terms:
                break

    return candidates
