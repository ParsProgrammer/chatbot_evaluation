# app/evaluator/validator.py
import json
import os
import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import List, Optional, Dict, Any, Tuple

import numpy as np


def _norm(s: str) -> str:
    return " ".join((s or "").lower().strip().split())


def _fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _load_intent_aliases() -> dict[str, List[str]]:
    raw = os.getenv("INTENT_ALIASES_JSON", "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return {}
        out: dict[str, List[str]] = {}
        for k, vals in obj.items():
            if isinstance(vals, list):
                out[_norm(str(k))] = [_norm(str(v)) for v in vals if _norm(str(v))]
        return out
    except Exception:
        return {}


def intent_matches(predicted: str, expected: str) -> bool:
    # backward compatible: bool only
    return intent_evaluate(predicted, expected)["passed"]


def intent_evaluate(predicted: str, expected: str) -> Dict[str, Any]:
    """
    Rich intent evaluation:
      returns {passed: bool, method: str, score: float}
    Env:
      INTENT_MATCH_MODE = exact | alias | prefix | fuzzy | hybrid (default exact)
      INTENT_FUZZY_THRESHOLD = 0.90
      INTENT_PREFIX_SEPARATOR = "."
    """
    p = _norm(predicted)
    e = _norm(expected)
    if not p or not e:
        return {"passed": False, "method": "empty", "score": 0.0}

    mode = os.getenv("INTENT_MATCH_MODE", "exact").strip().lower()
    thr = float(os.getenv("INTENT_FUZZY_THRESHOLD", "0.90"))
    sep = os.getenv("INTENT_PREFIX_SEPARATOR", ".")

    def exact() -> Tuple[bool, float]:
        ok = (p == e)
        return ok, (1.0 if ok else 0.0)

    def alias() -> Tuple[bool, float]:
        aliases = _load_intent_aliases()
        if p == e:
            return True, 1.0
        if e in aliases and p in aliases[e]:
            return True, 1.0
        if p in aliases and e in aliases[p]:
            return True, 1.0
        return False, 0.0

    def prefix() -> Tuple[bool, float]:
        if p == e:
            return True, 1.0
        if sep and (p.startswith(e + sep) or e.startswith(p + sep)):
            return True, 0.95
        return False, 0.0

    def fuzzy() -> Tuple[bool, float]:
        if p == e:
            return True, 1.0
        score = _fuzzy_ratio(p, e)
        return score >= thr, score

    strategies = [("exact", exact), ("alias", alias), ("prefix", prefix), ("fuzzy", fuzzy)]

    if mode in {"exact", "alias", "prefix", "fuzzy"}:
        for name, fn in strategies:
            if name == mode:
                ok, score = fn()
                return {"passed": ok, "method": name, "score": float(score)}
        return {"passed": False, "method": mode, "score": 0.0}

    # hybrid: pick the best passing method; else best score
    best = {"passed": False, "method": "none", "score": 0.0}
    for name, fn in strategies:
        ok, score = fn()
        if ok and (not best["passed"] or score > best["score"]):
            best = {"passed": True, "method": name, "score": float(score)}
        elif (not best["passed"]) and score > best["score"]:
            best = {"passed": False, "method": name, "score": float(score)}
    return best


# -------------------------
# Response evaluation
# -------------------------

_SENT_SPLIT_RE = re.compile(r"[.!?]\s+|\n+")


def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [t.strip() for t in _SENT_SPLIT_RE.split(text) if t.strip()]
    return parts if parts else [text]


# ---- Embedding-based semantic similarity (pretrained Transformer) ----

@lru_cache(maxsize=1)
def _get_embedder():
    """
    Pretrained Transformer embeddings via Sentence-Transformers.
    Suppresses HF Hub + Transformers warnings.
    """

    # --- Silence HuggingFace warnings/logging ---
    import os
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Silence transformers logging programmatically
    try:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    # Silence sentence-transformers logging
    try:
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    except Exception:
        pass

    from sentence_transformers import SentenceTransformer

    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    ).strip()

    return SentenceTransformer(model_name)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


@lru_cache(maxsize=4096)
def _embed_text(text: str) -> np.ndarray:
    model = _get_embedder()
    v = model.encode(text)
    return np.asarray(v, dtype=np.float32)


def _semantic_best_score(response_text: str, expected_text: str) -> float:
    """
    Best cosine similarity between expected_text and any sentence in response_text.
    Fallback: fuzzy ratio if embedding fails.
    """
    sentences = _split_sentences(response_text)
    if not sentences:
        return 0.0

    expected_text = (expected_text or "").strip()
    if not expected_text:
        return 0.0

    try:
        exp_vec = _embed_text(expected_text)
        best = 0.0
        for s in sentences:
            s = (s or "").strip()
            if not s:
                continue
            s_vec = _embed_text(s)
            best = max(best, _cosine_similarity(exp_vec, s_vec))
        return float(max(0.0, min(1.0, best)))
    except Exception:
        en = _norm(expected_text)
        best = 0.0
        for s in sentences:
            best = max(best, _fuzzy_ratio(en, _norm(s)))
        return float(max(0.0, min(1.0, best)))


def keyword_pass(response_text: str, keywords: list[str]) -> bool:
    # backward compatible: bool only
    return response_evaluate(response_text, keywords)["passed"]


def _short(text: str, n: int = 60) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= n:
        return t
    return t[: n - 1] + "…"


def response_evaluate(response_text: str, keywords: list[str]) -> Dict[str, Any]:
    """
    Rich response evaluation:
      returns { passed: bool, rule_hits: List[str], semantic_score: Optional[float] }

    Supported tokens (dataset unchanged; still list[str]):
      "__ALL__"                 -> require ALL positive checks to pass
      "re:<pattern>"            -> regex must match response
      "!<text>"                 -> fail if <text> appears
      "sim:<expected meaning>"  -> semantic similarity with default threshold (SIM_THRESHOLD, default 0.65)
      "sim>=0.75:<text>"        -> semantic similarity with explicit threshold
      (plain keyword)           -> substring match (existing behavior)

    NEW:
      - Always computes semantic_score for reporting when plain keywords exist
        (using joined keywords as the expected text).
      - Optional semantic fallback pass (fail->pass) when only plain keywords exist:
          RESPONSE_SEMANTIC_FALLBACK=1 (default 1)
    """
    if not keywords:
        return {"passed": True, "rule_hits": ["no_requirements"], "semantic_score": None}

    raw = response_text or ""
    r = _norm(raw)

    require_all = any(_norm(k) == "__all__" for k in keywords if _norm(k))
    sim_default = float(os.getenv("SIM_THRESHOLD", "0.65"))
    semantic_fallback = os.getenv("RESPONSE_SEMANTIC_FALLBACK", "1").strip().lower() not in {"0", "false", "no"}

    positives: List[bool] = []
    rule_hits: List[str] = []
    semantic_score: Optional[float] = None
    negatives_triggered = False
    saw_explicit_sim_rule = False

    plain_keywords: List[str] = []

    for k in keywords:
        if not _norm(k):
            continue
        kn = _norm(k)

        if kn == "__all__":
            continue

        if kn.startswith("!"):
            needle = _norm(kn[1:])
            if needle and needle in r:
                negatives_triggered = True
                rule_hits.append(f"neg_hit:{needle}")
            else:
                rule_hits.append(f"neg_ok:{needle}" if needle else "neg_ok:empty")
            continue

        if kn.startswith("re:"):
            pattern = k.strip()[3:]
            try:
                ok = re.search(pattern, raw, flags=re.IGNORECASE | re.MULTILINE) is not None
            except re.error:
                ok = False
            positives.append(ok)
            rule_hits.append(f"regex({_short(pattern)}):{'pass' if ok else 'fail'}")
            continue

        if kn.startswith("sim"):
            saw_explicit_sim_rule = True
            thr = sim_default
            expected_text = ""
            m = re.match(r"sim\s*>=\s*([0-9]*\.?[0-9]+)\s*:(.*)$", k.strip(), flags=re.IGNORECASE)
            if m:
                thr = float(m.group(1))
                expected_text = m.group(2).strip()
            else:
                if ":" in k:
                    expected_text = k.split(":", 1)[1].strip()

            if expected_text:
                score = _semantic_best_score(raw, expected_text)
                semantic_score = score if (semantic_score is None) else max(semantic_score, score)
                ok = score >= thr
                positives.append(ok)
                rule_hits.append(f"semantic({_short(expected_text)}):{'pass' if ok else 'fail'}@{thr}")
            else:
                positives.append(False)
                rule_hits.append("semantic(missing_expected):fail")
            continue

        # plain keyword
        plain_keywords.append(k.strip())
        ok = _norm(k) in r
        positives.append(ok)
        rule_hits.append(f"kw({_short(k)}):{'pass' if ok else 'fail'}")

    # If any negative rule triggered -> fail (but still report semantic score)
    if negatives_triggered:
        if semantic_score is None and plain_keywords:
            semantic_score = _semantic_best_score(raw, " ".join(plain_keywords))
        return {"passed": False, "rule_hits": rule_hits, "semantic_score": semantic_score}

    if not positives:
        # only negatives present and none triggered -> pass
        return {"passed": True, "rule_hits": rule_hits or ["only_negatives_ok"], "semantic_score": semantic_score}

    passed = all(positives) if require_all else any(positives)

    # Always compute semantic score for reporting if plain keywords exist and no explicit semantic computed
    if semantic_score is None and plain_keywords:
        semantic_score = _semantic_best_score(raw, " ".join(plain_keywords))

    # Optional semantic fallback when only plain keywords exist (no explicit sim rules)
    if (not passed) and semantic_fallback and (not saw_explicit_sim_rule) and plain_keywords and semantic_score is not None:
        if semantic_score >= sim_default:
            passed = True
            rule_hits.append(f"semantic_fallback:pass@{sim_default}")
        else:
            rule_hits.append(f"semantic_fallback:fail@{sim_default}")

    return {"passed": bool(passed), "rule_hits": rule_hits, "semantic_score": semantic_score}