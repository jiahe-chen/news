"""The Newsren: an agentic news intelligence and learning platform.

Technology, education, industry, and finance news, researched, verified,
connected, and explained by an editorially controlled multi-agent workflow.
"""

__version__ = "0.1.0"


def main() -> None:
    from newsren.cli import main as cli_main

    cli_main()
