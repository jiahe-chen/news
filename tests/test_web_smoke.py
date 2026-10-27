"""Crawl the demo site as each kind of reader: no server errors, no broken forms, no raw markers."""

from __future__ import annotations

import re
from collections import deque
from urllib.parse import urldefrag, urlsplit

import pytest

LINK = re.compile(r'(?:href|hx-get)="(/[^"]*)"')
SKIP_PREFIXES = ("/static/", "/logout", "/api/")
MAX_PAGES = 160


def crawl(client, start: str = "/") -> dict[str, int]:
    seen: dict[str, int] = {}
    queue: deque[str] = deque([start, "/about", "/sources", "/finance", "/ask", "/me", "/editor", "/analytics"])
    problems: list[str] = []
    while queue and len(seen) < MAX_PAGES:
        url = urldefrag(queue.popleft())[0]
        if url in seen or url.startswith(SKIP_PREFIXES) or not url:
            continue
        resp = client.get(url, follow_redirects=True)
        seen[url] = resp.status_code
        if resp.status_code >= 500:
            problems.append(f"{url}: HTTP {resp.status_code}")
            continue
        if "text/html" not in resp.headers.get("content-type", ""):
            continue
        html = resp.text
        if 'name="csrf_token" value=""' in html:
            problems.append(f"{url}: form with empty csrf_token")
        for artifact in ("{{", "{%", "Undefined", "Traceback"):
            if artifact in html:
                problems.append(f"{url}: template artifact {artifact!r}")
        for href in LINK.findall(html):
            path = urldefrag(href.replace("&amp;", "&"))[0]
            if urlsplit(path).path not in {urlsplit(u).path for u in seen}:
                queue.append(path)
    assert not problems, "\n".join(problems)
    return seen


@pytest.mark.parametrize("who", ["anonymous", "reader", "editor"])
def test_crawl_site(request, who):
    client = request.getfixturevalue(
        {"anonymous": "client", "reader": "reader_client", "editor": "editor_client"}[who]
    )
    seen = crawl(client)
    ok = [u for u, code in seen.items() if code == 200]
    assert len(ok) >= 25, f"crawl reached only {len(ok)} pages: {sorted(seen)[:40]}"
    # Key sections are reachable for everyone.
    for path in ("/", "/about", "/sources", "/finance", "/ask"):
        assert seen.get(path) == 200, (path, seen.get(path))
    if who == "editor":
        assert seen.get("/editor") == 200 and seen.get("/analytics") == 200
    if who == "reader":
        assert seen.get("/me") == 200
        assert seen.get("/editor") == 403
