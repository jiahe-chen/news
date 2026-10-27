"""The .env loader: BOM-tolerant, fills empty shell variables, never overrides real values."""

from __future__ import annotations

from newsren.config import load_dotenv


def test_dotenv_handles_bom_quotes_export_and_empty_shell_vars(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        '﻿NEWSREN_TEST_KEY="sk-ant-test"\nexport NEWSREN_TEST_MODE = claude\n# comment\nNEWSREN_TEST_SET=from-file\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("NEWSREN_TEST_KEY", raising=False)
    monkeypatch.setenv("NEWSREN_TEST_MODE", "")  # present but empty in the shell
    monkeypatch.setenv("NEWSREN_TEST_SET", "from-shell")
    load_dotenv(env)
    import os

    assert os.environ["NEWSREN_TEST_KEY"] == "sk-ant-test"
    assert os.environ["NEWSREN_TEST_MODE"] == "claude"
    assert os.environ["NEWSREN_TEST_SET"] == "from-shell"
