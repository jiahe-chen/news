"""Text utilities shared by ingestion, story intelligence, and offline agents.

Pure Python, no heavy NLP dependencies: tokenization, keyword scoring,
sentence splitting, extractive summaries, URL canonicalization, and SimHash
near-duplicate fingerprints.
"""

from __future__ import annotations

import hashlib
import html
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are aren't as at be because been before being
    below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down
    during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here
    here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
    let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves
    out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their
    theirs them themselves then there there's these they they'd they'll they're they've this those through
    to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's
    where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're
    you've your yours yourself yourselves said says say new news year years one two first also may might
    will just like get got make made many much via per according including report reports reported
    announced announces today week month could would percent mr ms dr inc ltd corp co company
    """.split()
)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-']*[A-Za-z0-9]|[A-Za-z]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_name",
    "gclid", "fbclid", "mc_cid", "mc_eid", "ref", "ref_src", "cmpid", "ito", "at_medium", "at_campaign",
}  # fmt: skip


# ---------------------------------------------------------------- cleaning


def strip_html(raw: str | None) -> str:
    if not raw:
        return ""
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>", "\n", text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    lines = [_WS.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def normalize_space(text: str | None) -> str:
    return _WS.sub(" ", text or "").strip()


def truncate(text: str | None, limit: int, ellipsis: str = "…") -> str:
    text = normalize_space(text)
    if len(text) <= limit:
        return text
    cut = text[: limit - len(ellipsis)]
    if " " in cut[limit // 2 :]:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(",;:- ") + ellipsis


# ---------------------------------------------------------------- tokens & keywords


def tokenize(text: str | None, keep_stopwords: bool = False) -> list[str]:
    words = [w.lower().strip("'-") for w in _WORD.findall(text or "")]
    if keep_stopwords:
        return [w for w in words if w]
    return [w for w in words if w and w not in STOPWORDS and len(w) > 1 and not w.isdigit()]


def ngrams(tokens: Sequence[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def term_frequencies(text: str, bigrams: bool = True) -> Counter[str]:
    toks = tokenize(text)
    counts = Counter(toks)
    if bigrams:
        counts.update(ngrams(toks, 2))
    return counts


def top_keywords(text: str, k: int = 8, doc_freq: dict[str, int] | None = None, n_docs: int = 1) -> list[str]:
    """Keywords by (optionally IDF-weighted) frequency; bigrams favoured slightly."""
    counts = term_frequencies(text)
    scored: list[tuple[float, str]] = []
    for term, tf in counts.items():
        idf = math.log((1 + n_docs) / (1 + doc_freq.get(term, 0))) + 1 if doc_freq else 1.0
        weight = 1.3 if " " in term else 1.0
        scored.append((tf * idf * weight, term))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out: list[str] = []
    for _, term in scored:
        if any(term in kept or kept in term for kept in out):
            continue
        out.append(term)
        if len(out) >= k:
            break
    return out


def tfidf_vectors(docs: Sequence[str]) -> list[dict[str, float]]:
    """L2-normalized TF-IDF vectors (unigrams + bigrams) for a small corpus."""
    tfs = [term_frequencies(d) for d in docs]
    df: Counter[str] = Counter()
    for tf in tfs:
        df.update(tf.keys())
    n = len(docs)
    vectors = []
    for tf in tfs:
        vec = {t: (1 + math.log(c)) * (math.log((1 + n) / (1 + df[t])) + 1) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({t: v / norm for t, v in vec.items()})
    return vectors


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------- sentences & summaries


def split_sentences(text: str | None) -> list[str]:
    text = normalize_space(text)
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if len(p.strip()) > 1]


def extractive_summary(text: str, max_sentences: int = 3, query: str | None = None) -> list[str]:
    """Pick the most representative sentences (optionally biased toward a query), in original order."""
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return sentences
    doc_tf = Counter(tokenize(text))
    query_terms = set(tokenize(query)) if query else set()
    scored = []
    for idx, sent in enumerate(sentences):
        toks = tokenize(sent)
        if not toks:
            continue
        score = sum(doc_tf[t] for t in toks) / (len(toks) ** 0.7)
        score += 3.0 * len(query_terms.intersection(toks))
        score *= 1.25 if idx == 0 else 1.0  # lede bias
        scored.append((score, idx))
    keep = sorted(idx for _, idx in sorted(scored, reverse=True)[:max_sentences])
    return [sentences[i] for i in keep]


def best_snippet(text: str, query: str, max_len: int = 280) -> str:
    """Sentence from ``text`` that best matches ``query`` (for citation snippets)."""
    sentences = split_sentences(text)
    if not sentences:
        return ""
    q = set(tokenize(query))
    best = max(sentences, key=lambda s: (len(q.intersection(tokenize(s))), -abs(len(s) - 160)))
    return truncate(best, max_len)


# ---------------------------------------------------------------- URLs, hashes, dates


def canonical_url(url: str) -> str:
    """Lower-case scheme/host, drop fragments, tracking params, and trailing slashes."""
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "https").lower()
    if scheme == "http":
        scheme = "https"
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host.endswith(":443") or host.endswith(":80"):
        host = host.rsplit(":", 1)[0]
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query) if k.lower() not in TRACKING_PARAMS))
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if len(path) > 1:
        path = path.rstrip("/")
    return urlunsplit((scheme, host, path, query, ""))


def domain_of(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    host = host.split("@")[-1].split(":")[0]
    return host[4:] if host.startswith("www.") else host


def content_hash(*parts: str | None) -> str:
    norm = " ".join(tokenize(" ".join(p or "" for p in parts), keep_stopwords=True))
    return hashlib.sha256(norm.encode()).hexdigest()


def simhash(text: str, bits: int = 64) -> int:
    """64-bit SimHash over word 3-shingles (fallback to unigrams for short text)."""
    toks = tokenize(text, keep_stopwords=True)
    features = ngrams(toks, 3) if len(toks) >= 6 else toks
    if not features:
        return 0
    weights = [0] * bits
    for feat, count in Counter(features).items():
        h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
        for i in range(bits):
            weights[i] += count if (h >> i) & 1 else -count
    value = 0
    for i, w in enumerate(weights):
        if w > 0:
            value |= 1 << i
    return value


def simhash_hex(text: str) -> str:
    return f"{simhash(text):016x}"


def hamming(a: int | str, b: int | str) -> int:
    ia = int(a, 16) if isinstance(a, str) else a
    ib = int(b, 16) if isinstance(b, str) else b
    return (ia ^ ib).bit_count()


def parse_datetime(value: str | datetime | None) -> datetime | None:
    """Parse RFC-822 / ISO-8601 strings into naive UTC datetimes."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        raw = value.strip()
        dt = None
        try:
            dt = parsedate_to_datetime(raw)
        except (TypeError, ValueError, IndexError):
            pass
        if dt is None:
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d %B %Y", "%B %d, %Y"):
                    try:
                        dt = datetime.strptime(raw, fmt)
                        break
                    except ValueError:
                        continue
        if dt is None:
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def iso(dt: datetime | None) -> str | None:
    return dt.replace(microsecond=0).isoformat() + "Z" if dt else None


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:max_len].rstrip("-") or "item"
