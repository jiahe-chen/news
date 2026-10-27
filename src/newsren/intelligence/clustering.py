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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsren.intelligence.entities import PLACE, entity_head, tag_article_entities
from newsren.models import Article, ArticleEntity, DocType, Entity, Story, TrustTier, utcnow
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
    """normalized key -> (display name, salience) for an article's non-place entities."""
    if not article.entities:
        tag_article_entities(session, article)
    return {
        link.entity.normalized: (link.entity.name, link.salience)
        for link in article.entities
        if link.entity.kind != PLACE
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


def _tfidf(docs: Sequence[str]) -> list[dict[str, float]]:
    """L2-normalized unigram TF-IDF vectors (sublinear tf) over light-stemmed text.

    Unigrams rather than ``textutil.tfidf_vectors``' unigrams + bigrams: news
    summaries of one story rarely share word pairs, and unmatched bigrams
    dilute the cosine between differently worded reports.
    """
    tfs = [Counter(tokenize(similarity_text(d))) for d in docs]
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
# Story state during a clustering pass
# --------------------------------------------------------------------------


@dataclass
class _StoryState:
    story: Story
    members: list[Article] = field(default_factory=list)  # non-duplicates, chronological
    _entities: set[str] | None = None

    @classmethod
    def load(cls, story: Story) -> _StoryState:
        members = [a for a in story.articles if a.duplicate_of_id is None]
        members.sort(key=lambda a: (article_time(a), a.id or 0))
        return cls(story=story, members=members)

    def recent(self) -> list[Article]:
        return self.members[-CENTROID_MEMBERS:]

    def centroid_text(self) -> str:
        return " ".join(article_doc(a) for a in self.recent())

    def entity_keys(self, session: Session) -> set[str]:
        if self._entities is None:
            totals: dict[str, float] = {}
            for art in self.recent():
                for key, weight in _match_weights(session, art).items():
                    totals[key] = totals.get(key, 0.0) + weight
            self._entities = _top_keys(totals, STORY_CORE_ENTITIES)
        return self._entities

    def add(self, article: Article) -> None:
        self.members.append(article)
        self.members.sort(key=lambda a: (article_time(a), a.id or 0))
        self._entities = None

    def eligible(self, when: datetime, window: timedelta) -> bool:
        return self.story.last_updated_at >= when - window and self.story.first_seen_at <= when + window


def _scores(
    article_text: str, article_keys: set[str], states: Sequence[_StoryState], session: Session
) -> list[float]:
    """Raw similarity of one article to each story (before the vertical adjustment)."""
    if not states:
        return []
    vectors = _tfidf([article_text, *(st.centroid_text() for st in states)])
    return [
        TEXT_WEIGHT * cosine(vectors[0], vectors[i + 1])
        + ENTITY_WEIGHT * jaccard(article_keys, st.entity_keys(session))
        for i, st in enumerate(states)
    ]


def title_anchors(session: Session, article: Article) -> set[str]:
    """Organization/product keys (or brand heads) whose names appear in the article's headline."""
    if not article.entities:
        tag_article_entities(session, article)
    title = (article.title or "").lower()
    anchors: set[str] = set()
    for link in article.entities:
        ent = link.entity
        if ent.kind == PLACE or not ent.name or ent.name.lower() not in title:
            continue
        head = entity_head(ent.name, ent.kind)
        anchors.add(head or ent.normalized)
    return anchors


def _anchor_df(session: Session, articles: Sequence[Article]) -> Counter[str]:
    df: Counter[str] = Counter()
    for art in articles:
        df.update(title_anchors(session, art))
    return df


def _article_keys(session: Session, article: Article) -> set[str]:
    return _top_keys(_match_weights(session, article), ARTICLE_ENTITY_LIMIT)


def match_score(session: Session, article: Article, story: Story) -> float:
    """Similarity of ``article`` to ``story`` as used by ``assign_stories`` (no vertical adjustment).

    Scored against this story alone, so IDF weights match an ``assign_stories``
    pass in which this story is the only candidate.
    """
    state = _StoryState.load(story)
    state.members = [m for m in state.members if m.id != article.id]
    return _scores(article_doc(article), _article_keys(session, article), [state], session)[0]


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

    corpus = [m for st in states for m in st.members] + ordered
    anchor_df = _anchor_df(session, corpus)
    max_anchor_df = max(2, int(ANCHOR_MAX_DF_SHARE * len(corpus)))
    changed: dict[int, Story] = {}
    for art in ordered:
        when = article_time(art)
        eligible = [st for st in states if st.eligible(when, window)]
        scores = _scores(article_doc(art), _article_keys(session, art), eligible, session)
        anchors = {a for a in title_anchors(session, art) if anchor_df[a] <= max_anchor_df}
        best: _StoryState | None = None
        best_margin = 0.0
        for st, score in zip(eligible, scores, strict=True):
            adjusted = score - (CROSS_VERTICAL_MARGIN if st.story.vertical != art.vertical else 0.0)
            needed = threshold
            if anchors and score >= ANCHOR_MIN_TEXT and st.story.vertical == art.vertical:
                member_anchors: set[str] = set()
                for member in st.members:
                    member_anchors |= title_anchors(session, member)
                if anchors & member_anchors:
                    needed = min(threshold, ANCHORED_THRESHOLD)
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
