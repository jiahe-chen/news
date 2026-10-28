"""Command-line interface: ``newsren <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from newsren.config import get_settings
from newsren.db import init_db, session_scope

DEMO_EDITOR = ("editor@newsren.example", "editor-demo-pass", "Demo Editor")
DEMO_READER = ("reader@newsren.example", "reader-demo-pass", "Demo Reader")


def _print(data) -> None:
    print(json.dumps(data, indent=2, default=str))


def cmd_init(args: argparse.Namespace) -> int:
    from newsren.ingest.sources import sync_sources

    init_db()
    with session_scope() as session:
        sources = sync_sources(session)
    print(f"Database ready at {get_settings().database_url}; {len(sources)} approved sources registered.")
    return 0


def _ensure_user(session, email: str, password: str, name: str, role: str):
    from sqlalchemy import select

    from newsren.models import User
    from newsren.users.auth import create_user

    user = session.scalars(select(User).where(User.email == email)).first()
    if user is None:
        user = create_user(session, email, password, name, role=role)
    return user


def _load_sample_earnings(session) -> int:
    """Build an Earnings Decoder report from a bundled public SEC EDGAR sample (no network)."""
    from importlib.resources import files

    from newsren.config import get_settings
    from newsren.finance.edgar import build_earnings_report, earnings_brief, get_or_create_company

    samples = files("newsren.finance") / "data" / "samples"
    submissions = json.loads((samples / "irmd_submissions.json").read_text())
    facts = json.loads((samples / "irmd_companyfacts.json").read_text())
    company = get_or_create_company(
        session,
        submissions["name"],
        ticker=submissions["tickers"][0],
        cik=submissions["cik"],
        sector=submissions.get("sicDescription"),
    )
    report = build_earnings_report(session, company, get_settings(), facts=facts, submissions=submissions)
    if report.brief_id is None:
        earnings_brief(session, report)
    return 1


def demo_signoff(session, editor) -> int:
    """The demo editor signs off drafts so the front page has content.

    Non-finance drafts are approved unless a check raised a blocker or a
    warning (info-level flags such as "high-impact claims" are fine). Finance
    drafts are approved only when every check passed, so flagged finance
    stories (performance figures, earnings numbers) stay in the review queue
    to show the human gate.
    """
    from newsren.editorial.workflow import approve, review_queue

    published = 0
    for brief in review_queue(session):
        failed = [c for c in brief.check_results if not c.get("passed")]
        serious = any(c.get("severity") in ("blocker", "warning") for c in failed)
        if (brief.vertical == "finance" and failed) or serious:
            continue
        approve(session, brief, editor, notes="Demo sign-off: no blocking issues in automated checks.")
        published += 1
    return published


def cmd_demo_signoff(args: argparse.Namespace) -> int:
    """Approve demo drafts after `newsren demo` (e.g. Claude-written drafts with info-level flags)."""
    from sqlalchemy import select

    from newsren.models import User

    init_db()
    with session_scope() as session:
        editor = session.scalars(select(User).where(User.email == DEMO_EDITOR[0])).first()
        if editor is None:
            print("Run `newsren demo` first (the demo editor account does not exist).", file=sys.stderr)
            return 1
        published = demo_signoff(session, editor)
    print(f"Demo editor approved {published} draft(s). Refresh the front page.")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Load fictional demo content, run the full workflow offline, and create demo accounts."""
    from sqlalchemy import select

    from newsren.agents.briefing import build_briefing, deliver_briefing
    from newsren.agents.monitor import run_monitoring
    from newsren.models import Audience, Story
    from newsren.pipeline import run_pipeline
    from newsren.users.service import add_interest, create_follow, set_preferences

    init_db()
    with session_scope() as session:
        report = run_pipeline(
            session, network=args.network, demo=True, max_briefs=args.max_briefs, monitor=False, briefings=False
        )
        earnings = _load_sample_earnings(session)
        editor = _ensure_user(session, *DEMO_EDITOR, role="editor")
        reader = _ensure_user(session, *DEMO_READER, role="reader")
        published = 0
        if not args.keep_queue:
            published = demo_signoff(session, editor)
        if not reader.interests:
            set_preferences(session, reader, audience=Audience.STUDENT, email_opt_in=True)
            add_interest(session, reader, "topic", "AI models for learning mathematics")
            add_interest(session, reader, "organization", "Lumen Labs")
            add_interest(session, reader, "topic", "semiconductor capacity")
            model_story = session.scalars(
                select(Story).where(Story.title.contains("Lumen-4")).order_by(Story.id)
            ).first()
            create_follow(
                session,
                reader,
                "Tell me when an independent benchmark is published for Lumen-4, "
                "not whenever someone writes another opinion article about it.",
                target_type="story" if model_story else "topic",
                story=model_story,
            )
            create_follow(
                session,
                reader,
                "Tell me about Corvane Semiconductor filings, earnings, or enforcement actions.",
                target_type="company",
                entity_name="Corvane Semiconductor",
                vertical="finance",
            )
        monitoring = run_monitoring(session)
        briefing = build_briefing(session, reader, force=True)
        if briefing is not None:
            deliver_briefing(session, briefing, reader)
    _print(
        {
            "pipeline": report.as_dict(),
            "sample_earnings_reports": earnings,
            "demo_published": published,
            "monitoring": monitoring,
        }
    )
    print(
        "\nDemo accounts:\n"
        f"  editor: {DEMO_EDITOR[0]} / {DEMO_EDITOR[1]}\n"
        f"  reader: {DEMO_READER[0]} / {DEMO_READER[1]}\n"
        "Run `newsren serve` and open http://localhost:8000"
    )
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from newsren.ingest.pipeline import ingest_all
    from newsren.ingest.sources import sync_sources

    init_db()
    with session_scope() as session:
        sync_sources(session)
        report = ingest_all(session, source_slugs=args.source or None)
    _print(report.__dict__)
    return 0 if not report.sources_failed or report.sources_ok else 1


def cmd_pipeline(args: argparse.Namespace) -> int:
    from newsren.pipeline import run_pipeline

    init_db()
    with session_scope() as session:
        report = run_pipeline(session, network=not args.no_network, max_briefs=args.max_briefs)
    _print(report.as_dict())
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    from newsren.agents.monitor import run_monitoring

    init_db()
    with session_scope() as session:
        _print(run_monitoring(session))
    return 0


def cmd_briefings(args: argparse.Namespace) -> int:
    from newsren.agents.briefing import run_briefings

    init_db()
    with session_scope() as session:
        _print(run_briefings(session))
    return 0


def cmd_finance(args: argparse.Namespace) -> int:
    from newsren.finance.service import refresh_finance

    init_db()
    with session_scope() as session:
        if args.ticker:
            from newsren.finance.edgar import (
                build_earnings_report,
                earnings_brief,
                get_or_create_company,
                lookup_cik,
            )

            settings = get_settings()
            cik = lookup_cik(args.ticker, settings)
            if not cik:
                print(f"Unknown ticker {args.ticker}", file=sys.stderr)
                return 1
            company = get_or_create_company(session, args.ticker.upper(), ticker=args.ticker.upper(), cik=cik)
            report = build_earnings_report(session, company, settings)
            brief = earnings_brief(session, report)
            print(f"Earnings report {report.fiscal_period} ({report.form}) -> brief #{brief.id} [{brief.status}]")
        _print(refresh_finance(session, network=args.network))
    return 0


def cmd_create_user(args: argparse.Namespace) -> int:
    from newsren.users.auth import create_user

    init_db()
    with session_scope() as session:
        user = create_user(
            session, args.email, args.password, args.name or args.email.split("@")[0], role=args.role
        )
        print(f"Created {user.role} {user.email} (id {user.id})")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    from newsren.metrics import all_metrics

    init_db()
    with session_scope() as session:
        _print(all_metrics(session))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    init_db()
    uvicorn.run("newsren.web.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="newsren", description="The Newsren — agentic news intelligence.")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the database and register approved sources").set_defaults(func=cmd_init)

    p = sub.add_parser("demo", help="load fictional demo content and run the workflow offline")
    p.add_argument("--network", action="store_true", help="also ingest live approved feeds")
    p.add_argument("--max-briefs", type=int, default=40)
    p.add_argument("--keep-queue", action="store_true", help="leave every draft in the editor review queue")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("ingest", help="fetch approved sources")
    p.add_argument("--source", action="append", help="limit to a source slug (repeatable)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("pipeline", help="run discover -> connect -> explain -> verify -> output")
    p.add_argument("--no-network", action="store_true", help="skip live ingestion")
    p.add_argument("--max-briefs", type=int, default=25)
    p.set_defaults(func=cmd_pipeline)

    sub.add_parser("demo-signoff", help="demo editor approves drafts without blocking issues").set_defaults(
        func=cmd_demo_signoff
    )

    sub.add_parser("monitor", help="check every active follow for material developments").set_defaults(
        func=cmd_monitor
    )
    sub.add_parser("briefings", help="build and deliver personal briefings").set_defaults(func=cmd_briefings)

    p = sub.add_parser("finance", help="refresh finance trackers")
    p.add_argument("--network", action="store_true", help="refresh EDGAR earnings data for tracked companies")
    p.add_argument("--ticker", help="build an Earnings Decoder report for one ticker (uses SEC EDGAR)")
    p.set_defaults(func=cmd_finance)

    p = sub.add_parser("create-user", help="create a reader or editor account")
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--name")
    p.add_argument("--role", choices=["reader", "editor", "admin"], default="reader")
    p.set_defaults(func=cmd_create_user)

    sub.add_parser(
        "metrics", help="print trust/usefulness/education/engagement/editorial/cost metrics"
    ).set_defaults(func=cmd_metrics)

    p = sub.add_parser("serve", help="run the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sys.exit(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    main()
