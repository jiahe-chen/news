"""Reader personalization: preferences, interests, follows, alerts, and feedback.

My Newsren (plan §2.1) builds briefings from a reader's interests and
preferred audience/depth; Follow This Story (§2.6) monitors a reader's
question over time and alerts only when an event meets the reader's
criteria. ``create_follow`` turns the reader's own words into those criteria:

    "Tell me when an independent benchmark is published for this model, not
    whenever someone writes another opinion article about it."

becomes ``event_types=["independent_evaluation", "benchmark"]``,
``require_independent=True`` and ``exclude_doc_types=["opinion"]``. Phrases in
a negated part of the question ("not ...", "rather than ...", "except ...")
never add event types; they exclude formats (opinion, blogs, preprints) or
tighten independence instead. Derivation is deterministic and rule-based so
readers can see and edit exactly what will trigger an alert; nuanced
conditions go into ``FollowCriteria.custom``, which the monitoring agent
judges against each candidate.

Functions flush but never commit; callers own the transaction.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsren.models import (
    Alert,
    Audience,
    Depth,
    DocType,
    EventType,
    Feedback,
    Follow,
    Interest,
    Story,
    User,
    Vertical,
    utcnow,
)
from newsren.schemas import FollowCriteria
from newsren.textutil import STOPWORDS, normalize_space, tokenize, top_keywords

BRIEFING_FREQUENCIES = ("daily", "weekly", "when_meaningful")
INTEREST_KINDS = ("topic", "question", "organization", "sector")
FOLLOW_TARGETS = ("topic", "story", "company", "sector")
ALERT_FEEDBACK = ("useful", "unwanted")
FEEDBACK_TARGETS = ("briefing", "answer", "brief", "learning", "impact", "evidence", "product", "alert")

MAX_INTEREST_LENGTH = 500
MAX_QUESTION_LENGTH = 2000
MAX_QUESTION_KEYWORDS = 8
MAX_FOLLOW_KEYWORDS = 20

# Words that phrase the request rather than describe the subject.
FILLER_WORDS = frozenset(
    """
    tell me notify notified notifying notification notifications alert alerts alerted alerting let know
    knowing whenever someone somebody anyone anybody something anything everything writes write writing
    written wrote another article articles piece pieces post posts posted posting coverage update updates
    updated keep track tracking follow following followed watch watching please want wants interested
    hear happens happen published publishes publish publishing every time times just only send ping
    remind story stories thing things lets i'd i'm files filed issues issued releases released comes
    gets appears appear
    """.split()
)

_NEGATION = re.compile(
    r"\b(?:not|no|never|except|excluding|exclude|without|ignore|skip|nothing about|rather than|instead of"
    r"|other than|don't|do not|doesn't|does not|won't)\b",
    re.IGNORECASE,
)
_CLAUSE_SPLIT = re.compile(r"[.;:!?,()\n]+|\s+but\s+|\s+-{1,2}\s+|\s+—\s+", re.IGNORECASE)


# --------------------------------------------------------------------------
# Preferences and interests (My Newsren)
# --------------------------------------------------------------------------


def _choice(value: Any, allowed: Iterable[str], label: str) -> str:
    normalized = str(value).strip().lower()
    allowed = tuple(allowed)
    if normalized not in allowed:
        raise ValueError(f"Unknown {label} {value!r}; choose one of {', '.join(allowed)}.")
    return normalized


def _vertical(value: str | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return _choice(value, (v.value for v in Vertical), "vertical")


def set_preferences(
    session: Session,
    user: User,
    audience: str | None = None,
    depth: str | None = None,
    email_opt_in: bool | str | None = None,
    briefing_frequency: str | None = None,
) -> User:
    """Update the reader's "Explain for" audience, technical depth, email opt-in, and briefing frequency.

    Arguments left as None are unchanged. Raises ``ValueError`` for values
    outside the vocabularies (``Audience``, ``Depth``, ``BRIEFING_FREQUENCIES``).
    """
    if audience is not None:
        user.audience = _choice(audience, (a.value for a in Audience), "audience")
    if depth is not None:
        user.depth = _choice(depth, (d.value for d in Depth), "depth")
    if email_opt_in is not None:
        # Form posts arrive as strings, and bool("false") is True.
        if isinstance(email_opt_in, str):
            user.email_opt_in = email_opt_in.strip().lower() in {"1", "true", "yes", "on"}
        else:
            user.email_opt_in = bool(email_opt_in)
    if briefing_frequency is not None:
        user.briefing_frequency = _choice(briefing_frequency, BRIEFING_FREQUENCIES, "briefing frequency")
    session.flush()
    return user


def add_interest(session: Session, user: User, kind: str, value: str, vertical: str | None = None) -> Interest:
    """Add a topic, question, organization, or sector to the reader's interests.

    Re-adding an existing interest (same kind, case-insensitive value) returns
    the existing row instead of a duplicate.
    """
    kind_value = _choice(kind, INTEREST_KINDS, "interest kind")
    text = normalize_space(value)
    if not text:
        raise ValueError("Describe the interest, e.g. a topic, a question, or an organization.")
    if len(text) > MAX_INTEREST_LENGTH:
        raise ValueError(f"Interests must be at most {MAX_INTEREST_LENGTH} characters.")
    vertical_value = _vertical(vertical)
    for existing in user.interests:
        if existing.kind == kind_value and existing.value.strip().lower() == text.lower():
            if vertical_value and not existing.vertical:
                existing.vertical = vertical_value
                session.flush()
            return existing
    interest = Interest(user=user, kind=kind_value, value=text, vertical=vertical_value)
    session.add(interest)
    session.flush()
    return interest


def remove_interest(session: Session, user: User, interest_id: int) -> bool:
    """Remove one of the reader's interests. Returns False if it does not exist or belongs to someone else."""
    interest = session.get(Interest, interest_id)
    if interest is None or interest.user_id != user.id:
        return False
    user.interests.remove(interest)  # delete-orphan cascade deletes the row
    session.flush()
    return True


# --------------------------------------------------------------------------
# Follow This Story: criteria and keyword derivation
# --------------------------------------------------------------------------

_E = EventType

# (pattern, event types added when the phrase appears un-negated)
_EVENT_RULES: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (
        re.compile(
            r"\bindependent(?:ly)?\s+(?:\w+\s+)?(?:benchmark\w*|evaluat\w*|test\w*|review\w*|audit\w*|assess\w*"
            r"|verif\w*|replicat\w*)|\bthird[- ]part(?:y|ies)\b|\bexternal (?:evaluation|audit|benchmark)s?\b",
            re.IGNORECASE,
        ),
        (_E.INDEPENDENT_EVALUATION, _E.BENCHMARK),
    ),
    (
        re.compile(
            r"\bbenchmark(?:s|ed|ing)?\b|\bevaluat(?:ion|ions|ed|es|ing)\b|\bleaderboards?\b", re.IGNORECASE
        ),
        (_E.BENCHMARK, _E.INDEPENDENT_EVALUATION),
    ),
    (
        re.compile(
            r"\bpeer[- ]?reviewed\b|\bpublished (?:study|studies|paper|papers|research|trial|evaluation)\b"
            r"|\bjournal (?:article|paper|publication)s?\b|\b(?:studies|study|paper|papers|preprints?)\b"
            r"|\brandomi[sz]ed\b",
            re.IGNORECASE,
        ),
        (_E.RESEARCH_PUBLICATION,),
    ),
    (
        re.compile(
            r"\bfilings?\b|\bfiles?\s+(?:a|an|its|their)\b|\b10-?[qk]s?\b|\b8-?ks?\b|\bs-1\b|\bproxy statement\b"
            r"|\bannual report\b|\bsec form\b",
            re.IGNORECASE,
        ),
        (_E.FILING,),
    ),
    (
        re.compile(r"\bearnings\b|\bquarterly (?:results|report|numbers)\b|\bresults call\b", re.IGNORECASE),
        (_E.EARNINGS,),
    ),
    (
        re.compile(
            r"\bfunding\b|\bfundraise\w*\b|\braises?\b|\braised\b|\braising\b|\bseries [a-f]\b"
            r"|\bseed (?:round|funding)\b|\bventure (?:round|capital)\b",
            re.IGNORECASE,
        ),
        (_E.FUNDING,),
    ),
    (
        re.compile(
            r"\benforcement\b|\bfines?\b(?![-\w])|\bfined\b|\bpenalt(?:y|ies)\b|\bsanction(?:s|ed)?\b"
            r"|\bcease[- ]and[- ]desist\b|\bcharg(?:ed|es) (?:with|against)\b|\bsettle(?:s|d|ment)?\b"
            r"|\bwashing\b",
            re.IGNORECASE,
        ),
        (_E.ENFORCEMENT,),
    ),
    (
        re.compile(
            r"\bregulat(?:ion|ions|ory|or|ors|e|es|ed)\b|\brules?\b|\brulemaking\b|\blegislation\b|\blaws?\b"
            r"|\bpolic(?:y|ies)\b|\bcurriculum\b|\bmandates?\b",
            re.IGNORECASE,
        ),
        (_E.REGULATION, _E.POLICY),
    ),
    (
        re.compile(r"\bcorrections?\b|\bcorrected\b|\bretract(?:s|ed|ion|ions)?\b", re.IGNORECASE),
        (_E.CORRECTION,),
    ),
    (
        re.compile(r"\blaunch(?:es|ed|ing)?\b|\bunveil(?:s|ed|ing)?\b|\bnow available\b", re.IGNORECASE),
        (_E.PRODUCT_LAUNCH,),
    ),
    (
        re.compile(
            r"\badopt(?:s|ed|ing|ion|ions)?\b|\bdeploy(?:s|ed|ing|ment|ments)?\b|\brolls?\s+out\b"
            r"|\brolled\s+out\b|\brollouts?\b",
            re.IGNORECASE,
        ),
        (_E.ADOPTION,),
    ),
    (
        re.compile(
            r"\bvulnerabilit(?:y|ies)\b|\bsecurity (?:advisory|advisories|flaw|flaws|patch)\b|\bcve\b"
            r"|\bbreach(?:es)?\b|\bexploit(?:s|ed)?\b",
            re.IGNORECASE,
        ),
        (_E.SECURITY_ADVISORY,),
    ),
]

_OPINION = re.compile(
    r"\bopinions?\b|\bop-?eds?\b|\bcommentary\b|\bcolumns?\b|\bcolumnists?\b|\beditorials?\b|\bhot takes?\b"
    r"|\bthink ?pieces?\b",
    re.IGNORECASE,
)
_BLOG = re.compile(r"\bblogs?\b|\bblog posts?\b|\bnewsletters?\b", re.IGNORECASE)
_PREPRINT = re.compile(r"\bpreprints?\b|\bnot (?:yet )?peer[- ]?reviewed\b|\bunreviewed\b", re.IGNORECASE)
# Not bare "journal": "the Wall Street Journal reports ..." is about a publisher, not peer review.
_PEER_REVIEWED = re.compile(
    r"\bpeer[- ]?reviewed\b|\bjournal (?:article|paper|publication)s?\b|\bpublished in (?:a|an|the) \w*\s?journal\b",
    re.IGNORECASE,
)
_PUBLISHED_STUDY = re.compile(
    r"\bpublished (?:study|studies|paper|papers|research|trial|evaluation)\b", re.IGNORECASE
)
_VENDOR = re.compile(
    r"\bvendors?\b|\bcompany(?:'s)? own\b|\bself-reported\b|\bpress releases?\b|\bmarketing\b"
    r"|\bcompany claims?\b|\bthe company says\b|\bpromotional\b",
    re.IGNORECASE,
)
_PRIMARY = re.compile(
    r"\bprimary sources?\b|\boriginal (?:source|record|document|filing)s?\b|\bofficial (?:filing|record|source|"
    r"announcement|document)s?\b",
    re.IGNORECASE,
)

PEER_REVIEW_NOTE = (
    "Only peer-reviewed or formally published research meets this follow; preprints and press releases "
    "about unpublished work do not."
)
PUBLISHED_STUDY_NOTE = (
    "Only a published study (with its methods and results available) meets this follow, not announcements "
    "or press coverage that describe results without the study."
)


def _split_polarity(question: str) -> tuple[str, str]:
    """Split a question into its affirmative and negated parts.

    Each clause (split on punctuation and "but") is affirmative up to its
    first negation cue and negated after it: "an independent benchmark, not
    whenever someone writes another opinion article" -> ("an independent
    benchmark", "whenever someone writes another opinion article").
    """
    positive: list[str] = []
    negative: list[str] = []
    for clause in _CLAUSE_SPLIT.split(question or ""):
        clause = clause.strip()
        if not clause:
            continue
        cue = _NEGATION.search(clause)
        if cue is None:
            positive.append(clause)
            continue
        if clause[: cue.start()].strip():
            positive.append(clause[: cue.start()].strip())
        if clause[cue.end() :].strip():
            negative.append(clause[cue.end() :].strip())
    return " . ".join(positive), " . ".join(negative)


def _ordered_unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(i for i in items if i))


def derive_criteria(question: str) -> FollowCriteria:
    """Notification criteria implied by the reader's question (rule-based, see module docstring)."""
    positive, negative = _split_polarity(question)
    event_types: list[str] = []
    for pattern, types in _EVENT_RULES:
        if pattern.search(positive):
            event_types.extend(str(t) for t in types)

    exclude = [DocType.OPINION.value]
    if _OPINION.search(positive) and not _OPINION.search(negative):
        # The reader explicitly asked for commentary, so do not filter it out.
        exclude.remove(DocType.OPINION.value)
        event_types.extend([EventType.OPINION.value, EventType.ANALYSIS.value])
    if _BLOG.search(negative):
        exclude.append(DocType.BLOG.value)
    if _PREPRINT.search(negative) or _PEER_REVIEWED.search(positive):
        exclude.append(DocType.RESEARCH_PREPRINT.value)

    require_independent = bool(_EVENT_RULES[0][0].search(positive)) or bool(_VENDOR.search(negative))
    require_primary = bool(_PRIMARY.search(positive))

    notes: list[str] = []
    if _PEER_REVIEWED.search(positive):
        notes.append(PEER_REVIEW_NOTE)
    elif _PUBLISHED_STUDY.search(positive):
        notes.append(PUBLISHED_STUDY_NOTE)

    return FollowCriteria(
        event_types=_ordered_unique(event_types),
        exclude_doc_types=_ordered_unique(exclude),
        require_independent=require_independent,
        require_primary=require_primary,
        custom=" ".join(notes),
    )


def _content_runs(text: str) -> list[list[str]]:
    """Maximal runs of adjacent content words (stopwords, filler, and numbers break a run)."""
    runs: list[list[str]] = []
    for clause in text.split(" . "):
        run: list[str] = []
        for token in tokenize(clause, keep_stopwords=True):
            if token in STOPWORDS or token in FILLER_WORDS or len(token) < 2 or token.isdigit():
                if run:
                    runs.append(run)
                run = []
            else:
                run.append(token)
        if run:
            runs.append(run)
    return runs


def derive_keywords(question: str, limit: int = MAX_QUESTION_KEYWORDS) -> list[str]:
    """Subject keywords for a follow question: phrases first, then remaining terms, filler removed.

    Only the affirmative part of the question is used (what the reader wants,
    not what they excluded). Phrases are built only from words that are
    adjacent in the question, so "benchmark ... for this model" never becomes
    "benchmark model": a run of two or three content words is one phrase
    ("wall street journal"); longer runs are split into their best
    ``top_keywords`` bigrams, and leftover words follow as single keywords.
    """
    positive, _negative = _split_polarity(question)
    runs = _content_runs(positive)
    tokens = [t for run in runs for t in run]
    if not tokens:
        return []
    ranking = {term: i for i, term in enumerate(top_keywords(" ".join(tokens), k=4 * len(tokens)))}
    phrases: list[str] = []
    covered: set[str] = set()
    for run in runs:
        if 2 <= len(run) <= 3:
            phrases.append(" ".join(run))
            covered.update(run)
            continue
        bigrams = [f"{a} {b}" for a, b in zip(run, run[1:], strict=False)]
        for bigram in sorted(bigrams, key=lambda b: ranking.get(b, len(ranking))):
            words = bigram.split()
            if not covered.intersection(words):
                phrases.append(bigram)
                covered.update(words)
    counts = Counter(tokens)
    first_seen = {t: i for i, t in reversed(list(enumerate(tokens)))}
    singles = sorted((t for t in counts if t not in covered), key=lambda t: (-counts[t], first_seen[t]))
    return _ordered_unique(phrases + singles)[:limit]


def describe_criteria(criteria: FollowCriteria | dict[str, Any]) -> list[str]:
    """Plain-language lines describing what will (and will not) trigger an alert, for the reader's UI."""
    c = criteria if isinstance(criteria, FollowCriteria) else FollowCriteria.model_validate(criteria or {})
    lines: list[str] = []
    if c.event_types:
        lines.append("Alerts only for: " + ", ".join(t.replace("_", " ") for t in c.event_types) + ".")
    else:
        lines.append("Alerts for any material development.")
    if c.require_independent:
        lines.append("Requires an independent (non-vendor) source.")
    if c.require_primary:
        lines.append("Requires a primary source such as a filing, paper, or official release.")
    if c.exclude_doc_types:
        lines.append("Ignores: " + ", ".join(t.replace("_", " ") for t in c.exclude_doc_types) + " pieces.")
    if c.custom:
        lines.append("Also checks: " + c.custom)
    return lines


# --------------------------------------------------------------------------
# Follows
# --------------------------------------------------------------------------


def create_follow(
    session: Session,
    user: User,
    question: str,
    *,
    target_type: str = "topic",
    story: Story | None = None,
    entity_name: str | None = None,
    vertical: str | None = None,
    criteria: FollowCriteria | None = None,
) -> Follow:
    """Follow a question, story, company, or sector over time.

    Keywords come from the question (``derive_keywords``), plus the entity
    name and, for story follows, the story's own keywords. Criteria come from
    the question's phrasing (``derive_criteria``) unless ``criteria`` is given,
    which overrides derivation entirely. A ``story`` makes the follow a story
    follow and an ``entity_name`` alone makes it a company follow. Following
    the same question and target again reactivates and returns the existing
    follow instead of creating a duplicate (duplicate follows would mean
    duplicate alerts).
    """
    text = normalize_space(question)
    if not text:
        raise ValueError("Write the question you want The Newsren to follow.")
    if len(text) > MAX_QUESTION_LENGTH:
        raise ValueError(f"Follow questions must be at most {MAX_QUESTION_LENGTH} characters.")
    target = _choice(target_type or "topic", FOLLOW_TARGETS, "follow target")
    entity = normalize_space(entity_name) or None
    if target == "topic" and story is not None:
        target = "story"
    elif target == "topic" and entity:
        target = "company"
    if target == "story" and story is None:
        raise ValueError("A story follow needs the story to follow.")
    vertical_value = _vertical(vertical) or (story.vertical if story is not None else None)

    if criteria is None:
        criteria_model = derive_criteria(text)
    elif isinstance(criteria, FollowCriteria):
        criteria_model = criteria
    else:
        criteria_model = FollowCriteria.model_validate(criteria)

    keywords: list[str] = []
    if entity:
        keywords.append(entity.lower())
    keywords.extend(derive_keywords(text))
    if story is not None:
        keywords.extend(str(k).strip().lower() for k in (story.keywords or []) if str(k).strip())
    keywords = _ordered_unique(keywords)[:MAX_FOLLOW_KEYWORDS]

    story_id = story.id if story is not None else None
    for existing in user.follows:
        if (
            existing.question.strip().lower() == text.lower()
            and existing.target_type == target
            and existing.story_id == story_id
            and (existing.entity_name or None) == entity
        ):
            existing.active = True
            existing.keywords = keywords
            existing.criteria = criteria_model.model_dump()
            existing.vertical = vertical_value or existing.vertical
            session.flush()
            return existing

    follow = Follow(
        user=user,
        question=text,
        target_type=target,
        story_id=story_id,
        entity_name=entity,
        vertical=vertical_value,
        keywords=keywords,
        criteria=criteria_model.model_dump(),
        active=True,
    )
    session.add(follow)
    session.flush()
    return follow


def deactivate_follow(session: Session, follow: Follow | User, follow_id: int | None = None) -> Follow | None:
    """Stop monitoring a follow (its alerts are kept).

    Call as ``deactivate_follow(session, follow)`` or, mirroring
    ``remove_interest``, ``deactivate_follow(session, user, follow_id)``; the
    second form only touches the user's own follow and returns None when there
    is no such follow.
    """
    if isinstance(follow, User):
        if follow_id is None:
            raise TypeError("deactivate_follow(session, user, follow_id) needs a follow_id")
        target = session.get(Follow, follow_id)
        if target is None or target.user_id != follow.id:
            return None
    else:
        target = follow
    target.active = False
    session.flush()
    return target


def user_follows(session: Session, user: User, active_only: bool = True) -> list[Follow]:
    """The reader's follows, newest first."""
    query = select(Follow).where(Follow.user_id == user.id)
    if active_only:
        query = query.where(Follow.active.is_(True))
    return list(session.scalars(query.order_by(Follow.created_at.desc(), Follow.id.desc())))


# --------------------------------------------------------------------------
# Alerts
# --------------------------------------------------------------------------


def user_alerts(session: Session, user: User, unread_only: bool = False, limit: int | None = None) -> list[Alert]:
    """The reader's alerts, newest first (optionally only unread ones)."""
    query = select(Alert).where(Alert.user_id == user.id)
    if unread_only:
        query = query.where(Alert.read_at.is_(None))
    query = query.order_by(Alert.created_at.desc(), Alert.id.desc())
    if limit is not None:
        query = query.limit(limit)
    return list(session.scalars(query))


def unread_alert_count(session: Session, user: User) -> int:
    """Number of unread alerts (for navigation badges)."""
    return int(
        session.scalar(select(func.count(Alert.id)).where(Alert.user_id == user.id, Alert.read_at.is_(None))) or 0
    )


def mark_alert(session: Session, alert: Alert, feedback: str | None = None, read: bool = True) -> Alert:
    """Mark an alert read/unread and optionally record the reader's verdict.

    ``feedback`` is "useful" or "unwanted" (feeds the useful-vs-unwanted alert
    metric); None or an empty string leaves earlier feedback unchanged.
    ``read=False`` marks the alert unread again.
    """
    if feedback is not None and str(feedback).strip():
        alert.feedback = _choice(feedback, ALERT_FEEDBACK, "alert feedback")
    if read:
        if alert.read_at is None:
            alert.read_at = utcnow()
    else:
        alert.read_at = None
    session.flush()
    return alert


# --------------------------------------------------------------------------
# Reader feedback (usefulness metric)
# --------------------------------------------------------------------------


def record_feedback(
    session: Session,
    user: User | None,
    target_type: str,
    target_id: int,
    *,
    answered_question: bool | None = None,
    saved_time: bool | None = None,
    rating: int | None = None,
    comment: str | None = None,
) -> Feedback:
    """Record whether a briefing/answer/brief answered the reader's question and saved them time.

    A signed-in reader rating the same item again updates their earlier row,
    so each reader counts once per item in the usefulness metric. ``rating``
    is 1-5.
    """
    target = _choice(target_type, FEEDBACK_TARGETS, "feedback target")
    if rating is not None:
        rating = int(rating)
        if not 1 <= rating <= 5:
            raise ValueError("Ratings run from 1 to 5.")
    comment_text = normalize_space(comment)[:2000] or None
    row: Feedback | None = None
    if user is not None:
        row = session.scalars(
            select(Feedback).where(
                Feedback.user_id == user.id, Feedback.target_type == target, Feedback.target_id == int(target_id)
            )
        ).first()
    if row is None:
        row = Feedback(user_id=user.id if user is not None else None, target_type=target, target_id=int(target_id))
        session.add(row)
    if answered_question is not None:
        row.answered_question = bool(answered_question)
    if saved_time is not None:
        row.saved_time = bool(saved_time)
    if rating is not None:
        row.rating = rating
    if comment_text is not None:
        row.comment = comment_text
    session.flush()
    return row


def interests_by_kind(interests: Sequence[Interest]) -> dict[str, list[Interest]]:
    """Group interests by kind (topic/question/organization/sector) in that order, for display."""
    grouped: dict[str, list[Interest]] = {kind: [] for kind in INTEREST_KINDS}
    for interest in interests:
        grouped.setdefault(interest.kind, []).append(interest)
    return grouped
