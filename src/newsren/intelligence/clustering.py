"""Incremental story clustering and cross-story connections (Connect).

``assign_stories`` groups related articles into evolving ``Story`` rows
instead of treating every article as an isolated item:

* Candidates are stories whose activity window overlaps the article's time
  (``last_updated_at`` within ``window_days`` of the article).
* Similarity = 0.7 x TF-IDF cosine (article title + summary vs. the story's
  centroid text: title + summary of its most recent members; unigrams with
  light plural stemming) + 0.3 x Jaccard overlap of salient entities
  (places excluded: a shared country is weak evidence of a shared story;
  multi-word names also contribute their brand head, so "Corvane" matches
  "Corvane Semiconductor").
* Same vertical is preferred: a cross-vertical match must clear the
  threshold by ``CROSS_VERTICAL_MARGIN``.
* Headline anchors: a distinctive organization/product named in both the
  article's headline and a member headline lowers the bar to
  ``ANCHORED_THRESHOLD``. A publisher naming itself ("NIST" in NIST News,
  "Federal Reserve" in the Fed's feed) is never an anchor (``title_anchors``)
  and never links its own items; only another publisher's coverage that
  names a company may anchor to the company's own release. Named only in
  its body ("The NCSC is encouraging ..."), the publisher is a byline and
  not an entity for matching at all.
* Publisher boilerplate is not evidence. Per source (with at least
  ``BOILERPLATE_MIN_ITEMS`` items in the pass), words and entity keys found
  in at least ``BOILERPLATE_SHARE`` of its items, and rarely in other
  publishers' items, are dropped from that source's similarity text and
  entity keys ("files Form 10-Q (quarterly report)", "Federal Reserve Board
  announces"). A headline a source repeats for different items (a programme
  or series name such as "Tech Life") is not used either, nor a summary
  sentence it repeats ("This is today's edition of The Download ...").
* Releases (press releases, regulatory notices, filings, advisories,
  policy documents) from one publisher each announce a distinct action:
  two of them share a story only when they name a shared subject, never
  when they name different ones (an enforcement action against one
  respondent and another against a different respondent), and otherwise
  only with ``SAME_PUBLISHER_MARGIN`` of extra textual similarity.
* SEC filings (``doc_type`` filing with a ``cik`` in ``Article.extra``) only
  cluster by company: a filing joins a story holding filings of the same
  CIK (never another CIK) or, by the normal rules, a story whose members
  name the company; other articles join a filing story by the normal rules
  only when they name the filer.
* Near-duplicates (``duplicate_of_id``) never seed stories; they inherit
  their original's story and do not count as a story change.
* A new story gets a provisional title (the most representative member
  headline); ``timeline.refresh_story`` later writes the agent's neutral title.

``related_stories`` links stories across verticals by shared entities and
keywords (e.g. an AI model's technology story and the education story about
schools adopting it).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsren.intelligence.entities import (
    GAZETTEER,
    ORG,
    PLACE,
    entity_head,
    is_generic_name,
    normalize_entity,
    tag_article_entities,
)
from newsren.models import Article, ArticleEntity, DocType, Entity, Source, Story, TrustTier, utcnow
from newsren.textutil import cosine, jaccard, normalize_space, tfidf_vectors, tokenize, top_keywords, truncate

TEXT_WEIGHT = 0.7
ENTITY_WEIGHT = 0.3
CROSS_VERTICAL_MARGIN = 0.15
CENTROID_MEMBERS = 8  # recent members that define a story's centroid text
ARTICLE_ENTITY_LIMIT = 10
STORY_CORE_ENTITIES = 8  # a story's defining entities, compared with an article's
STORY_ENTITY_LIMIT = 15  # entities compared between stories (related_stories)
KEY_ENTITY_LIMIT = 10
KEYWORD_LIMIT = 8
ACTIVE_MEMBER_COUNT = 3  # members needed for a story to count as "active" rather than "developing"
MIN_RELATED_SCORE = 0.05
# Headline anchors: a distinctive organization/product named in both the
# article's headline and a member headline ("Lumen Tutor" / "Lumen Labs
# unveils Lumen-4") is strong evidence of one evolving story, so the match
# threshold drops to ANCHORED_THRESHOLD. Entities named in more than
# ANCHOR_MAX_DF_SHARE of the window's articles (big platforms, regulators)
# never anchor, and some textual overlap is still required.
ANCHORED_THRESHOLD = 0.17
ANCHOR_MIN_TEXT = 0.08
ANCHOR_MAX_DF_SHARE = 0.25
RELATED_CANDIDATE_LIMIT = 300
# Per-source boilerplate: in a source with at least BOILERPLATE_MIN_ITEMS items
# in the pass, a word or entity key found in BOILERPLATE_SHARE of them (and in
# at least BOILERPLATE_MIN_COUNT) is the publisher's template, not evidence of a
# shared story, unless other publishers use it about as often (then it is a
# story the source covers heavily: other publishers' items may contain it at
# most BOILERPLATE_OUTSIDE_RATIO times as often as this source's).
BOILERPLATE_MIN_ITEMS = 5
BOILERPLATE_SHARE = 0.4
BOILERPLATE_MIN_COUNT = 3
BOILERPLATE_OUTSIDE_RATIO = 0.25
# A summary sentence (of at least REPEATED_SENTENCE_MIN_WORDS words) that a source
# repeats in REPEATED_SENTENCE_MIN_ITEMS items is a template line, not content.
REPEATED_SENTENCE_MIN_WORDS = 6
REPEATED_SENTENCE_MIN_ITEMS = 3
# Releases from one publisher without a shared named subject must clear the
# threshold by this much (their shared vocabulary is the publisher's, not the story's).
SAME_PUBLISHER_MARGIN = 0.15
RELEASE_DOC_TYPES = frozenset(
    {DocType.PRESS_RELEASE, DocType.REGULATORY, DocType.FILING, DocType.ADVISORY, DocType.POLICY}
)

_TITLE_LABEL = re.compile(
    r"^\s*(?:opinion|analysis|exclusive|breaking|update[d]?|correction|clarification|explainer|commentary|"
    r"editorial|op-ed|watch|live)\s*[:|\-–—]\s*",
    re.IGNORECASE,
)

# Headlines built on a possibility make poor neutral story titles.
_SPECULATIVE = re.compile(r"\b(?:could|may|might|would)\b|\?", re.IGNORECASE)


# --------------------------------------------------------------------------
# Article helpers
# --------------------------------------------------------------------------


def article_time(article: Article) -> datetime:
    """When the article entered the story: publication time, else retrieval time."""
    return article.published_at or article.retrieved_at or utcnow()


def article_doc(article: Article) -> str:
    """Text used for story similarity: title + summary (or the start of the body)."""
    summary = normalize_space(article.summary) or truncate(article.content, 500)
    return f"{article.title}. {summary}" if summary else article.title


_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'’\-]*")


def _stem(word: str) -> str:
    w = word.lower()
    if w.endswith(("'s", "’s")):
        w = w[:-2]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es") and w[:-2].endswith(("s", "x", "ch", "sh")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def similarity_text(text: str) -> str:
    """Lower-cased text with light plural stemming ("schools" -> "school") so TF-IDF matches word forms."""
    return _WORD.sub(lambda m: _stem(m.group()), text)


def _entity_weights(session: Session, article: Article) -> dict[str, tuple[str, float]]:
    """normalized key -> (display name, salience) for an article's non-place, non-generic entities."""
    if not article.entities:
        tag_article_entities(session, article)
    return {
        link.entity.normalized: (link.entity.name, link.salience)
        for link in article.entities
        if link.entity.kind != PLACE and not is_generic_name(link.entity.name)
    }


def _with_heads(entries: Iterable[tuple[str, str, str, float]]) -> dict[str, float]:
    """Matching keys from (key, name, kind, weight): entity keys plus brand heads.

    "Corvane Semiconductor" also yields "corvane", so it matches an article
    that only says "Corvane".
    """
    out: dict[str, float] = {}
    for key, name, kind, weight in entries:
        if kind == PLACE:
            continue
        out[key] = max(out.get(key, 0.0), weight)
        head = entity_head(name, kind)
        if head and head != key:
            out[head] = max(out.get(head, 0.0), weight)
    return out


def _match_weights(session: Session, article: Article) -> dict[str, float]:
    if not article.entities:
        tag_article_entities(session, article)
    return _with_heads(
        (link.entity.normalized, link.entity.name, link.entity.kind, link.salience) for link in article.entities
    )


def _doc_tokens(text: str) -> list[str]:
    """Light-stemmed unigram tokens of a similarity text."""
    return tokenize(similarity_text(text))


def _tfidf(docs: Sequence[Sequence[str]]) -> list[dict[str, float]]:
    """L2-normalized unigram TF-IDF vectors (sublinear tf) over token lists (``_doc_tokens``).

    Unigrams rather than ``textutil.tfidf_vectors``' unigrams + bigrams: news
    summaries of one story rarely share word pairs, and unmatched bigrams
    dilute the cosine between differently worded reports.
    """
    tfs = [Counter(d) for d in docs]
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


def _top_keys(weights: dict[str, float], limit: int) -> set[str]:
    return {k for k, _ in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]}


def _root_original(session: Session, article: Article) -> Article | None:
    """Follow ``duplicate_of_id`` links to the original article (None if it is missing)."""
    seen: set[int] = set()
    current = article
    while current.duplicate_of_id is not None:
        if current.id in seen:
            return None
        seen.add(current.id)
        nxt = session.get(Article, current.duplicate_of_id)
        if nxt is None:
            return None
        current = nxt
    return current


def clean_title(title: str) -> str:
    """Headline without format labels ("Opinion:", "Exclusive:" ...) or trailing exclamation marks."""
    cleaned = _TITLE_LABEL.sub("", normalize_space(title)).rstrip("!").strip()
    return cleaned or normalize_space(title)


def best_member_title(articles: Sequence[Article]) -> str:
    """Most representative, most neutral member headline (a provisional story title).

    Scores each headline by TF-IDF similarity to the story's combined text,
    preferring reporting over commentary and independent/primary sources over
    vendor copy; ties go to the earliest article.
    """
    members = [a for a in articles if a.duplicate_of_id is None] or list(articles)
    if not members:
        return ""
    if len(members) == 1:
        return truncate(clean_title(members[0].title), 500)
    ordered = sorted(members, key=lambda a: (article_time(a), a.id or 0))
    centroid = " ".join(article_doc(a) for a in ordered)
    vectors = tfidf_vectors([a.title for a in ordered] + [centroid])
    best_score, best = float("-inf"), ordered[0]
    for i, art in enumerate(ordered):
        score = cosine(vectors[i], vectors[-1])
        if art.doc_type == DocType.OPINION or art.trust_tier == TrustTier.COMMENTARY:
            score -= 0.3
        elif art.trust_tier == TrustTier.VENDOR or art.doc_type == DocType.PRESS_RELEASE:
            score -= 0.05
        else:
            score += 0.05
        if _TITLE_LABEL.match(art.title or ""):
            score -= 0.2
        if _SPECULATIVE.search(art.title or ""):
            score -= 0.15
        if len(tokenize(art.title)) < 3:
            score -= 0.1
        if score > best_score + 1e-9:
            best_score, best = score, art
    return truncate(clean_title(best.title), 500)


# --------------------------------------------------------------------------
# Publishers, self-names, and SEC filings
# --------------------------------------------------------------------------

_SECOND_LEVEL = frozenset({"co", "com", "gov", "ac", "org", "net", "edu"})
_NAME_SPLIT = re.compile(r"[^a-z0-9]+")
# EDGAR company names carry a state of incorporation: "AMERICAN TOWER CORP /MA/", "RUSH ENTERPRISES INC \TX\".
_EDGAR_STATE = re.compile(r"\s*[/\\][A-Za-z]{1,3}[/\\]?\s*$")

# Canonical gazetteer key -> normalized keys of all its names ("federal reserve" -> {"fed", "federal reserve board", ...}).
_GAZETTEER_NAMES: dict[str, frozenset[str]] = {
    normalize_entity(canonical): frozenset(normalize_entity(n) for n in (canonical, *aliases))
    for canonical, _kind, aliases in GAZETTEER
}


def _host(source: Source | None) -> str:
    if source is None:
        return ""
    for url in (source.homepage, source.url):
        host = urlparse(url or "").hostname or ""
        if host:
            return host.lower().removeprefix("www.")
    return ""


def _publisher_key(source: Source | None) -> str:
    """The organization behind a source: its registrable domain ("sec.gov" for SEC press releases and EDGAR)."""
    labels = [label for label in _host(source).split(".") if label]
    if not labels:
        return f"source:{source.id}" if source is not None else ""
    if len(labels) >= 3 and labels[-2] in _SECOND_LEVEL and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _name_words(text: str) -> list[str]:
    return [w for w in _NAME_SPLIT.split(text.lower()) if w]


def _contains_phrase(words: Sequence[str], phrase: Sequence[str]) -> bool:
    n = len(phrase)
    return n > 0 and any(list(words[i : i + n]) == list(phrase) for i in range(len(words) - n + 1))


def _is_publisher_name(source: Source | None, entity_key: str) -> bool:
    """True when the entity is the publisher itself: "NIST" for NIST News, "SEC" for SEC press releases.

    The entity's key (or any gazetteer name of it) appears as words in the
    source name, or spelled out as a label of the source's domain
    ("federalreserve.gov").
    """
    if source is None or not entity_key:
        return False
    source_words = _name_words(source.name or "")
    labels = {label.replace("-", "") for label in _host(source).split(".")[:-1]}
    for key in _GAZETTEER_NAMES.get(entity_key, frozenset()) | {entity_key}:
        words = _name_words(key)
        if _contains_phrase(source_words, words) or "".join(words) in labels:
            return True
    return False


def _filing_cik(article: Article) -> str | None:
    """The filer's SEC CIK for an EDGAR filing (``doc_type`` filing with ``extra["cik"]``), else None."""
    if article.doc_type != DocType.FILING:
        return None
    cik = str((article.extra or {}).get("cik") or "").strip().lstrip("0")
    return cik or None


def _company_names(article: Article) -> tuple[str, str | None]:
    """A filing's company as (normalized name, distinctive brand head or None); ("", None) if unknown."""
    company = _EDGAR_STATE.sub("", str((article.extra or {}).get("company") or "")).strip()
    if not company:
        return "", None
    key = " ".join(_name_words(normalize_entity(company)))
    head = entity_head(company, ORG)
    return key, (head if head and head != key else None)


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def _split_sentences(text: str) -> list[str]:
    return [x for x in _SENTENCE_END.split(text) if x.strip()]


def _sentence_key(sentence: str) -> str:
    """Normalized sentence for repeat detection ('' for short ones, which repeat by chance)."""
    words = _name_words(sentence)
    return " ".join(words) if len(words) >= REPEATED_SENTENCE_MIN_WORDS else ""


# --------------------------------------------------------------------------
# Corpus statistics for one clustering pass
# --------------------------------------------------------------------------


class _Corpus:
    """Per-source boilerplate, repeated headlines, and cached similarity features for one pass.

    Statistics come from the pass's articles (members of the candidate
    stories plus the articles being clustered), so they describe what each
    source published around this time.
    """

    def __init__(self, session: Session, articles: Iterable[Article]) -> None:
        self.session = session
        items = list({id(a): a for a in articles if a.duplicate_of_id is None}.values())
        self._raw_tokens: dict[int, set[str]] = {}
        self._tokens: dict[int, list[str]] = {}
        self._keys: dict[int, dict[str, float]] = {}
        self._anchors: dict[tuple[int, bool, bool], set[str]] = {}
        self._words: dict[int, list[str]] = {}
        self._own_names: dict[tuple[int, str], bool] = {}

        by_source: dict[int, list[Article]] = {}
        for art in items:
            by_source.setdefault(art.source_id, []).append(art)
        self.publisher: dict[int, str] = {sid: _publisher_key(arts[0].source) for sid, arts in by_source.items()}

        # Headlines a source repeats for different items name a programme or series, not a story.
        self.series_titles: dict[int, set[str]] = {}
        for sid, arts in by_source.items():
            titles = Counter(self._title_key(a) for a in arts)
            self.series_titles[sid] = {t for t, c in titles.items() if t and c >= 2}

        # Summary sentences a source repeats across items ("This is today's edition of The Download ...").
        self.boiler_sentences: dict[int, set[str]] = {}
        for sid, arts in by_source.items():
            counts: Counter[str] = Counter()
            for art in arts:
                summary = normalize_space(art.summary) or truncate(art.content, 500)
                counts.update({k for x in _split_sentences(summary) if (k := _sentence_key(x))})
            self.boiler_sentences[sid] = {k for k, c in counts.items() if c >= REPEATED_SENTENCE_MIN_ITEMS}

        token_df: dict[int, Counter[str]] = {}
        key_df: dict[int, Counter[str]] = {}
        for sid, arts in by_source.items():
            tdf: Counter[str] = Counter()
            kdf: Counter[str] = Counter()
            for art in arts:
                tdf.update(self._raw(art))
                kdf.update(_match_weights(session, art).keys())
            token_df[sid], key_df[sid] = tdf, kdf
        self.boiler_tokens = self._boilerplate(by_source, token_df)
        self.boiler_keys = self._boilerplate(by_source, key_df)

    @staticmethod
    def _title_key(article: Article) -> str:
        return " ".join(_name_words(clean_title(article.title or "")))

    def _boilerplate(
        self, by_source: dict[int, list[Article]], df: dict[int, Counter[str]]
    ) -> dict[int, set[str]]:
        per_publisher: dict[str, Counter[str]] = {}
        for sid, counts in df.items():
            per_publisher.setdefault(self.publisher[sid], Counter()).update(counts)
        total: Counter[str] = Counter()
        for counts in per_publisher.values():
            total.update(counts)
        out: dict[int, set[str]] = {}
        for sid, arts in by_source.items():
            n = len(arts)
            if n < BOILERPLATE_MIN_ITEMS:
                out[sid] = set()
                continue
            floor = max(BOILERPLATE_MIN_COUNT, BOILERPLATE_SHARE * n)
            own = per_publisher[self.publisher[sid]]
            out[sid] = {
                term
                for term, count in df[sid].items()
                if count >= floor and total[term] - own[term] <= BOILERPLATE_OUTSIDE_RATIO * count
            }
        return out

    def is_series(self, article: Article) -> bool:
        return self._title_key(article) in self.series_titles.get(article.source_id, set())

    def _text(self, article: Article) -> str:
        summary = normalize_space(article.summary) or truncate(article.content, 500)
        repeated = self.boiler_sentences.get(article.source_id, set())
        if repeated:
            summary = " ".join(x for x in _split_sentences(summary) if _sentence_key(x) not in repeated)
        if self.is_series(article):
            return summary  # the repeated headline says nothing about this item
        return f"{article.title}. {summary}" if summary else article.title

    def _raw(self, article: Article) -> set[str]:
        key = id(article)
        if key not in self._raw_tokens:
            self._raw_tokens[key] = set(_doc_tokens(article_doc(article)))
        return self._raw_tokens[key]

    def tokens(self, article: Article) -> list[str]:
        """Similarity tokens without the source's boilerplate (or a repeated series headline)."""
        key = id(article)
        if key not in self._tokens:
            boiler = self.boiler_tokens.get(article.source_id, set())
            self._tokens[key] = [t for t in _doc_tokens(self._text(article)) if t not in boiler]
        return self._tokens[key]

    def weights(self, article: Article) -> dict[str, float]:
        """Entity matching weights without boilerplate keys, a series name, or the publisher as byline.

        A publisher naming itself only in the body ("The NCSC is encouraging
        UK organisations ...") is a byline; named in the headline ("Kestrel
        Semiconductor announces ...") it is the subject, which other
        publishers' coverage may share (``_scores`` still ignores it between
        the publisher's own items).
        """
        key = id(article)
        if key not in self._keys:
            if not article.entities:
                tag_article_entities(self.session, article)
            title = (article.title or "").lower()
            series = _name_words(title) if self.is_series(article) else []
            entries = [
                (link.entity.normalized, link.entity.name, link.entity.kind, link.salience)
                for link in article.entities
                if not is_generic_name(link.entity.name)
                and not (
                    self.is_own_name(article, link.entity.normalized) and link.entity.name.lower() not in title
                )
                and not (series and _contains_phrase(series, _name_words(link.entity.normalized)))
            ]
            boiler = self.boiler_keys.get(article.source_id, set())
            self._keys[key] = {k: w for k, w in _with_heads(entries).items() if k not in boiler}
        return self._keys[key]

    def is_own_name(self, article: Article, key: str) -> bool:
        """Whether ``key`` names the article's own publisher (cached ``_is_publisher_name``)."""
        cache_key = (article.source_id, key)
        if cache_key not in self._own_names:
            self._own_names[cache_key] = _is_publisher_name(article.source, key)
        return self._own_names[cache_key]

    def article_keys(self, article: Article) -> set[str]:
        """Top entity keys used to match ``article``: its publisher's own name never counts."""
        weights = {k: w for k, w in self.weights(article).items() if not self.is_own_name(article, k)}
        return _top_keys(weights, ARTICLE_ENTITY_LIMIT)

    def anchors(self, article: Article, include_own: bool = False) -> set[str]:
        """Headline anchors minus the source's boilerplate keys; none for a repeated series headline.

        The publisher's own name is left out unless ``include_own`` (matching
        another publisher's coverage of a company's own announcement).
        """
        return self._headline(article, with_heads=False, include_own=include_own)

    def subjects(self, article: Article) -> set[str]:
        """Named subjects of a headline: its anchors plus brand heads of body entities it names."""
        return self._headline(article, with_heads=True, include_own=False)

    def _headline(self, article: Article, with_heads: bool, include_own: bool) -> set[str]:
        key = (id(article), with_heads, include_own)
        if key not in self._anchors:
            if self.is_series(article):
                self._anchors[key] = set()
            else:
                boiler = self.boiler_keys.get(article.source_id, set())
                names = _headline_names(self.session, article, with_heads, include_own)
                self._anchors[key] = {a for a in names if a not in boiler}
        return self._anchors[key]

    def words(self, article: Article) -> list[str]:
        key = id(article)
        if key not in self._words:
            self._words[key] = _name_words(f"{article.title or ''} {article.summary or ''}")
        return self._words[key]

    def names_company(self, article: Article, filings: Iterable[Article]) -> bool:
        """Whether ``article`` names the company behind any of ``filings``.

        A multi-word company name may appear anywhere in the headline or
        summary; a one-word name or brand head ("PayPal") must be one of the
        article's extracted entities, so "no rush" never names Rush Enterprises.
        """
        words = self.words(article)
        keys = {" ".join(_name_words(k)) for k in _match_weights(self.session, article)}
        for filing in filings:
            name, head = _company_names(filing)
            if not name:
                continue
            if name in keys or (head and head in keys):
                return True
            if " " in name and _contains_phrase(words, name.split()):
                return True
        return False

    def publisher_of(self, article: Article) -> str:
        if article.source_id not in self.publisher:
            self.publisher[article.source_id] = _publisher_key(article.source)
        return self.publisher[article.source_id]


# --------------------------------------------------------------------------
# Story state during a clustering pass
# --------------------------------------------------------------------------


@dataclass
class _StoryState:
    story: Story
    members: list[Article] = field(default_factory=list)  # non-duplicates, chronological
    _entities: set[str] | None = None
    _centroid: list[str] | None = None

    @classmethod
    def load(cls, story: Story) -> _StoryState:
        members = [a for a in story.articles if a.duplicate_of_id is None]
        members.sort(key=lambda a: (article_time(a), a.id or 0))
        return cls(story=story, members=members)

    def recent(self) -> list[Article]:
        return self.members[-CENTROID_MEMBERS:]

    def centroid_tokens(self, corpus: _Corpus) -> list[str]:
        if self._centroid is None:
            self._centroid = [t for art in self.recent() for t in corpus.tokens(art)]
        return self._centroid

    def entity_keys(self, corpus: _Corpus) -> set[str]:
        if self._entities is None:
            totals: dict[str, float] = {}
            for art in self.recent():
                for key, weight in corpus.weights(art).items():
                    totals[key] = totals.get(key, 0.0) + weight
            self._entities = _top_keys(totals, STORY_CORE_ENTITIES)
        return self._entities

    def filings(self) -> list[Article]:
        return [m for m in self.members if _filing_cik(m)]

    def ciks(self) -> set[str]:
        return {cik for m in self.members if (cik := _filing_cik(m))}

    def add(self, article: Article) -> None:
        self.members.append(article)
        self.members.sort(key=lambda a: (article_time(a), a.id or 0))
        self._entities = None
        self._centroid = None

    def eligible(self, when: datetime, window: timedelta) -> bool:
        return self.story.last_updated_at >= when - window and self.story.first_seen_at <= when + window


def _scores(article: Article, states: Sequence[_StoryState], corpus: _Corpus) -> list[float]:
    """Raw similarity of one article to each story (before the vertical adjustment)."""
    if not states:
        return []
    vectors = _tfidf([corpus.tokens(article), *(st.centroid_tokens(corpus) for st in states)])
    keys = corpus.article_keys(article)
    scores = []
    for i, st in enumerate(states):
        # The article's own publisher, named by the story's members, is no evidence for it.
        story_keys = {k for k in st.entity_keys(corpus) if not corpus.is_own_name(article, k)}
        scores.append(TEXT_WEIGHT * cosine(vectors[0], vectors[i + 1]) + ENTITY_WEIGHT * jaccard(keys, story_keys))
    return scores


def _headline_names(session: Session, article: Article, with_heads: bool, include_own: bool = False) -> set[str]:
    if not article.entities:
        tag_article_entities(session, article)
    title = (article.title or "").lower()
    words = set(_name_words(title)) if with_heads else set()
    names: set[str] = set()
    for link in article.entities:
        ent = link.entity
        if ent.kind == PLACE or not ent.name or is_generic_name(ent.name):
            continue
        if not include_own and _is_publisher_name(article.source, ent.normalized):
            continue
        head = entity_head(ent.name, ent.kind)
        if ent.name.lower() in title:
            names.add(head or ent.normalized)
        elif head and head in words:
            names.add(head)  # "StepWise" in the headline, "StepWise Learning" in the body
    return names


def title_anchors(session: Session, article: Article) -> set[str]:
    """Organization/product keys (or brand heads) whose names appear in the article's headline.

    The publisher naming itself ("NIST" in a NIST News headline) is not an anchor.
    """
    return _headline_names(session, article, with_heads=False)


def _filing_compatible(corpus: _Corpus, article: Article, state: _StoryState) -> bool:
    """SEC filings cluster by company: same CIK, or coverage that names the filer."""
    cik = _filing_cik(article)
    story_ciks = state.ciks()
    if cik is not None:
        if story_ciks:
            return story_ciks == {cik}
        return any(corpus.names_company(member, [article]) for member in state.members)
    if story_ciks:
        return corpus.names_company(article, state.filings())
    return True


def match_score(session: Session, article: Article, story: Story) -> float:
    """Similarity of ``article`` to ``story`` as used by ``assign_stories`` (no vertical adjustment).

    Scored against this story alone, so IDF weights (and per-source
    boilerplate) match an ``assign_stories`` pass in which this story is the
    only candidate.
    """
    state = _StoryState.load(story)
    state.members = [m for m in state.members if m.id != article.id]
    corpus = _Corpus(session, [*state.members, article])
    return _scores(article, [state], corpus)[0]


def _refresh_story_stats(session: Session, state: _StoryState, window_days: int) -> None:
    story, members = state.story, state.members
    if not members:
        return
    times = [article_time(a) for a in members]
    story.first_seen_at = min(times)
    story.last_updated_at = max(times)
    story.keywords = top_keywords(" ".join(article_doc(a) for a in members), k=KEYWORD_LIMIT)
    totals: dict[str, float] = {}
    names: dict[str, str] = {}
    for art in members:
        for key, (name, salience) in _entity_weights(session, art).items():
            totals[key] = totals.get(key, 0.0) + salience
            names.setdefault(key, name)
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:KEY_ENTITY_LIMIT]
    story.key_entities = [names[k] for k, _ in ranked]
    if not story.what_happened_first:
        # Still provisional: the story-intelligence agent has not titled it yet.
        story.title = best_member_title(members) or story.title
    if story.last_updated_at < utcnow() - timedelta(days=window_days):
        story.status = "dormant"
    else:
        story.status = "active" if len(members) >= ACTIVE_MEMBER_COUNT else "developing"


def _needed_score(
    corpus: _Corpus,
    article: Article,
    anchors: set[str],
    state: _StoryState,
    score: float,
    threshold: float,
    anchor_ok: Callable[[str], bool],
) -> float | None:
    """The score ``article`` must reach to join ``state`` (None: it may not join)."""
    if not _filing_compatible(corpus, article, state):
        return None
    cik = _filing_cik(article)
    if cik is not None and cik in state.ciks():
        return min(threshold, ANCHORED_THRESHOLD)  # the same company's filings
    # A publisher's own name never links its own items ("NIST" in two NIST News
    # headlines), but independent coverage naming a company still anchors to
    # the company's own announcement.
    publisher = corpus.publisher_of(article)
    own_open = {a for a in corpus.anchors(article, include_own=True) if anchor_ok(a)}
    shared: set[str] = set()
    for member in state.members:
        if corpus.publisher_of(member) == publisher:
            shared |= anchors & corpus.anchors(member)
        else:
            shared |= own_open & corpus.anchors(member, include_own=True)
    if article.doc_type in RELEASE_DOC_TYPES:
        peers = [
            m for m in state.members if m.doc_type in RELEASE_DOC_TYPES and corpus.publisher_of(m) == publisher
        ]
        if peers:
            # A subject named in one headline that is salient on the other side
            # ("StepWise" in a clarification of the StepWise study release).
            own = {a for a in corpus.subjects(article) if anchor_ok(a)}
            peer_subjects = {a for m in peers for a in corpus.subjects(m) if anchor_ok(a)}
            member_subjects = {a for m in state.members for a in corpus.subjects(m)}
            subjects = (own & (member_subjects | state.entity_keys(corpus))) | (
                peer_subjects & corpus.article_keys(article)
            )
            if not subjects:
                if own and peer_subjects:
                    return None  # one publisher's releases about different named subjects
                return threshold + SAME_PUBLISHER_MARGIN
            shared = shared or subjects
    if shared and score >= ANCHOR_MIN_TEXT and state.story.vertical == article.vertical:
        return min(threshold, ANCHORED_THRESHOLD)
    return threshold


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def assign_stories(
    session: Session,
    articles: Iterable[Article] | None = None,
    window_days: int = 21,
    threshold: float = 0.28,
) -> list[Story]:
    """Attach unassigned articles to existing stories or start new ones.

    ``articles`` defaults to every article without a story. Non-duplicate
    articles are processed in chronological order and join the best-scoring
    candidate story if the score (minus ``CROSS_VERTICAL_MARGIN`` for a story
    in another vertical) reaches ``threshold``; otherwise they start a new
    story. Duplicates join their original's story (clustering the original
    first if needed). Returns the stories that gained non-duplicate members
    (new stories included), in the order they changed.
    """
    if articles is None:
        pending = list(session.scalars(select(Article).where(Article.story_id.is_(None))))
    else:
        pending = [a for a in articles if a.story_id is None and a.story is None]
    if not pending:
        return []

    singles: dict[int, Article] = {}
    duplicates: list[Article] = []
    for art in pending:
        if art.duplicate_of_id is None:
            singles[art.id] = art
            continue
        duplicates.append(art)
        root = _root_original(session, art)
        if root is None:
            singles[art.id] = art  # dangling duplicate link: cluster it on its own merits
        elif root.story_id is None and root.story is None:
            singles[root.id] = root
    ordered = sorted(singles.values(), key=lambda a: (article_time(a), a.id or 0))

    window = timedelta(days=window_days)
    states: list[_StoryState] = []
    if ordered:
        lo = article_time(ordered[0]) - window
        hi = article_time(ordered[-1]) + window
        candidates = session.scalars(
            select(Story).where(Story.last_updated_at >= lo, Story.first_seen_at <= hi)
        ).all()
        states = [_StoryState.load(s) for s in candidates]

    pool = [m for st in states for m in st.members] + ordered
    corpus = _Corpus(session, pool)
    anchor_df: Counter[str] = Counter()
    for art in pool:
        anchor_df.update(corpus.anchors(art, include_own=True))
    max_anchor_df = max(2, int(ANCHOR_MAX_DF_SHARE * len(pool)))

    def anchor_ok(key: str) -> bool:
        return anchor_df[key] <= max_anchor_df

    floor = min(threshold, ANCHORED_THRESHOLD)
    changed: dict[int, Story] = {}
    for art in ordered:
        when = article_time(art)
        eligible = [st for st in states if st.eligible(when, window)]
        scores = _scores(art, eligible, corpus)
        anchors = {a for a in corpus.anchors(art) if anchor_ok(a)}
        best: _StoryState | None = None
        best_margin = 0.0
        for st, score in zip(eligible, scores, strict=True):
            adjusted = score - (CROSS_VERTICAL_MARGIN if st.story.vertical != art.vertical else 0.0)
            if adjusted < floor:
                continue  # below any bar _needed_score can set
            needed = _needed_score(corpus, art, anchors, st, score, threshold, anchor_ok)
            if needed is None:
                continue
            margin = adjusted - needed
            if margin >= 0 and (best is None or margin > best_margin):
                best, best_margin = st, margin
        if best is None:
            story = Story(
                title=truncate(clean_title(art.title), 500),
                vertical=art.vertical,
                first_seen_at=when,
                last_updated_at=when,
                status="developing",
                is_demo=art.is_demo,
                disputed=[],
                evidence_changes=[],
                key_entities=[],
                keywords=[],
            )
            session.add(story)
            session.flush()
            best = _StoryState(story=story)
            states.append(best)
        art.story = best.story
        best.add(art)
        _refresh_story_stats(session, best, window_days)
        changed.setdefault(best.story.id, best.story)

    for dup in duplicates:
        if dup.story is not None:
            continue
        root = _root_original(session, dup)
        if root is not None and root.story is not None:
            dup.story = root.story
    session.flush()
    return list(changed.values())


def _story_entity_weights(session: Session, story_ids: Sequence[int]) -> dict[int, dict[str, float]]:
    """story id -> {matching key: summed salience} over non-duplicate members (places excluded, heads added)."""
    if not story_ids:
        return {}
    rows = session.execute(
        select(Article.story_id, Entity.normalized, Entity.name, Entity.kind, func.sum(ArticleEntity.salience))
        .join(ArticleEntity, ArticleEntity.article_id == Article.id)
        .join(Entity, Entity.id == ArticleEntity.entity_id)
        .where(Article.story_id.in_(story_ids), Article.duplicate_of_id.is_(None), Entity.kind != PLACE)
        .group_by(Article.story_id, Entity.id)
    ).all()
    grouped: dict[int, list[tuple[str, str, str, float]]] = {}
    for story_id, key, name, kind, weight in rows:
        grouped.setdefault(story_id, []).append((key, name, kind, float(weight or 0.0)))
    return {story_id: _with_heads(entries) for story_id, entries in grouped.items()}


def _keyword_tokens(story: Story) -> set[str]:
    keywords = story.keywords or top_keywords(
        " ".join(article_doc(a) for a in story.articles if a.duplicate_of_id is None), k=KEYWORD_LIMIT
    )
    return set(tokenize(" ".join(keywords)))


def related_stories(session: Session, story: Story, limit: int = 5) -> list[Story]:
    """Other stories (any vertical) connected to ``story`` by shared entities and keywords.

    Score = 0.6 x Jaccard of the stories' top entities + 0.4 x Jaccard of
    keyword tokens; best first, ties broken by recency. Excludes ``story``.
    """
    if story.id is None:
        return []
    own = _story_entity_weights(session, [story.id]).get(story.id, {})
    own_keys = _top_keys(own, STORY_ENTITY_LIMIT)
    own_tokens = _keyword_tokens(story)
    if not own_keys and not own_tokens:
        return []

    candidate_ids: set[int] = set()
    if own_keys:
        # Stories sharing an entity, or an entity whose name starts with a shared brand head.
        name_match = Entity.normalized.in_(own_keys)
        for head in (k for k in own_keys if " " not in k):
            name_match = name_match | Entity.normalized.like(f"{head} %")
        candidate_ids.update(
            session.scalars(
                select(Article.story_id)
                .join(ArticleEntity, ArticleEntity.article_id == Article.id)
                .join(Entity, Entity.id == ArticleEntity.entity_id)
                .where(name_match, Article.story_id.is_not(None), Article.story_id != story.id)
                .distinct()
            )
        )
    candidate_ids.update(
        session.scalars(
            select(Story.id)
            .where(Story.id != story.id)
            .order_by(Story.last_updated_at.desc())
            .limit(RELATED_CANDIDATE_LIMIT)
        )
    )
    if not candidate_ids:
        return []
    others = session.scalars(select(Story).where(Story.id.in_(candidate_ids))).all()
    weights = _story_entity_weights(session, [s.id for s in others])

    scored: list[tuple[float, datetime, Story]] = []
    for other in others:
        keys = _top_keys(weights.get(other.id, {}), STORY_ENTITY_LIMIT)
        score = 0.6 * jaccard(own_keys, keys) + 0.4 * jaccard(own_tokens, _keyword_tokens(other))
        if score >= MIN_RELATED_SCORE:
            scored.append((score, other.last_updated_at, other))
    scored.sort(key=lambda x: (-x[0], -x[1].timestamp(), x[2].id))
    return [s for _, _, s in scored[:limit]]
