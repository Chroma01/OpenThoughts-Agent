#!/usr/bin/env python3
"""DEPRECATED SHIM — re-homed to ``data/opencode_literals_to_sft/``.

This module (the bug-ledger #2 serve-parity rebuild for the densemixer A/B SFT experiment)
moved to reusable data-preprocessing code under ``data/opencode_literals_to_sft/``. It is kept
here as a thin re-export so existing references — including
``python -m scripts.harbor.literal_traces_to_opencode_sft`` — keep working.

Prefer the new path:
  python -m data.opencode_literals_to_sft --source_repo <repo> [--target_repo <repo>]
"""

from data.opencode_literals_to_sft.literals_to_sft import (  # noqa: F401
    build_row,
    convert,
    leading_turns,
    main,
    parse_tool_calls,
    parse_tools,
    recover_system_content,
    resolve_tokenizer_ref,
    strip_tool_calls,
)

if __name__ == "__main__":
    main()
