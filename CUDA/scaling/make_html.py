#!/usr/bin/env python3
# make_html.py — Convert scaling/scaling.md into a styled HTML report.
#
# Usage:
#   python3 scaling/make_html.py
#   python3 scaling/make_html.py --input scaling/scaling.md --output scaling/scaling.html

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


def protect_code_spans(s: str) -> tuple[str, list[str]]:
    codes: list[str] = []

    def repl(m: re.Match[str]) -> str:
        codes.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00CODE{len(codes) - 1}\x00"

    return re.sub(r"`([^`]*)`", repl, s), codes


def restore_code_spans(s: str, codes: list[str]) -> str:
    for i, code in enumerate(codes):
        s = s.replace(f"\x00CODE{i}\x00", code)
    return s


def inline_md(s: str) -> str:
    s = s.rstrip()
    protected, codes = protect_code_spans(s)
    escaped = html.escape(protected)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return restore_code_spans(escaped, codes)


def is_separator_row(line: str) -> bool:
    if not line.strip().startswith("|"):
        return False
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-+:?", c) for c in cells)


def split_table_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render_table(lines: list[str], start: int) -> tuple[str, int]:
    headers = split_table_row(lines[start])

    rows: list[list[str]] = []
    i = start + 2
    while i < len(lines):
        if not lines[i].strip().startswith("|"):
            break
        rows.append(split_table_row(lines[i]))
        i += 1

    out: list[str] = []
    out.append('<div class="table-wrap">')
    out.append("<table>")
    out.append("<thead><tr>")
    for h in headers:
        out.append(f"<th>{inline_md(h)}</th>")
    out.append("</tr></thead>")
    out.append("<tbody>")

    mlups_col = headers.index("MLUPS") if "MLUPS" in headers else -1

    for row_idx, row in enumerate(rows):
        cls = ' class="best"' if row_idx == 0 else ""
        out.append(f"<tr{cls}>")
        for col_idx, cell in enumerate(row):
            rendered = inline_md(cell)
            if row_idx == 0 and col_idx == mlups_col:
                rendered = f'<span class="metric">{rendered}</span>'
            out.append(f"<td>{rendered}</td>")
        out.append("</tr>")

    out.append("</tbody>")
    out.append("</table>")
    out.append("</div>")

    return "\n".join(out), i


def render_summary(lines: list[str]) -> str:
    cleaned = []
    for line in lines:
        s = line.strip()
        if s.startswith(">"):
            s = s[1:].strip()
        if s:
            cleaned.append(s)

    title = "Best CUDA results"
    cards: list[tuple[str, list[str]]] = []

    current_title: str | None = None
    current_body: list[str] = []

    for s in cleaned:
        if s.startswith("## "):
            title = s[3:].strip()
            continue

        m = re.match(r"\*\*(.+?):\*\*\s*(.*)", s)
        if m:
            if current_title is not None:
                cards.append((current_title, current_body))
            current_title = m.group(1).strip()
            rest = m.group(2).strip()
            current_body = [rest] if rest else []
        else:
            current_body.append(s)

    if current_title is not None:
        cards.append((current_title, current_body))

    out: list[str] = []
    out.append('<section class="summary">')
    out.append(f"<h2>{inline_md(title)}</h2>")
    out.append('<div class="summary-grid">')

    for card_title, body_lines in cards:
        body_html = "<br>\n".join(inline_md(x) for x in body_lines)
        out.append('<div class="summary-item">')
        out.append(f"<h3>{inline_md(card_title)}</h3>")
        out.append(f"<p>{body_html}</p>")
        out.append("</div>")

    out.append("</div>")
    out.append("</section>")
    return "\n".join(out)


def md_to_body(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("# "):
            out.append(f"<h1>{inline_md(stripped[2:].strip())}</h1>")
            i += 1
            continue

        if stripped.startswith("## "):
            out.append(f"<h2>{inline_md(stripped[3:].strip())}</h2>")
            i += 1
            continue

        if stripped.startswith(">"):
            block: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(lines[i])
                i += 1
            out.append(render_summary(block))
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and is_separator_row(lines[i + 1]):
            table_html, i = render_table(lines, i)
            out.append(table_html)
            continue

        if stripped.startswith("_") and stripped.endswith("_"):
            note = stripped[1:-1]
            out.append(f'<p class="note">{inline_md(note)}</p>')
        else:
            out.append(f"<p>{inline_md(stripped)}</p>")

        i += 1

    return "\n".join(out)


def wrap_html(body: str) -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>LBM CUDA Performance Results</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --border: #e5e7eb;
      --accent: #7c3aed;
      --accent-soft: #ede9fe;
      --success: #059669;
      --code-bg: #f3f4f6;
      --table-head: #f5f3ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 32px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
    }
    main { max-width: 1280px; margin: 0 auto; }
    h1 { margin: 0 0 24px; font-size: 2.2rem; letter-spacing: -0.02em; }
    h2 { margin-top: 36px; margin-bottom: 14px; font-size: 1.35rem; border-bottom: 2px solid var(--border); padding-bottom: 8px; }
    .summary {
      background: var(--card);
      border: 1px solid var(--border);
      border-left: 6px solid var(--accent);
      border-radius: 14px;
      padding: 22px 26px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
      margin-bottom: 28px;
    }
    .summary h2 { margin-top: 0; border-bottom: none; padding-bottom: 0; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
    .summary-item { background: var(--accent-soft); border-radius: 12px; padding: 16px; }
    .summary-item h3 { margin: 0 0 8px; font-size: 1rem; color: #5b21b6; }
    .summary-item p { margin: 0; }
    code {
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 1px 6px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 0.92em;
    }
    .table-wrap {
      overflow-x: auto;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
      margin-bottom: 18px;
    }
    table { width: 100%; border-collapse: collapse; min-width: 1080px; }
    thead { background: var(--table-head); }
    th, td { padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }
    th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3), th:nth-child(4), td:nth-child(4) { text-align: left; }
    tbody tr:hover { background: #f9fafb; }
    tbody tr.best { background: #ecfdf5; }
    tbody tr.best:hover { background: #d1fae5; }
    .metric { font-weight: 700; color: var(--success); }
    .note { color: var(--muted); font-style: italic; margin: 8px 0 26px; }
    footer { margin-top: 40px; color: var(--muted); font-size: 0.95rem; text-align: center; }
    @media print {
      body { background: white; padding: 16px; }
      .summary, .table-wrap { box-shadow: none; }
      table { font-size: 0.82rem; }
    }
  </style>
</head>
<body>
<main>
""" + body + """
  <footer>Generated from LBM CUDA performance logs.</footer>
</main>
</body>
</html>
"""


def main() -> int:
    here = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", type=Path, default=here / "scaling.md")
    ap.add_argument("--output", "-o", type=Path, default=here / "scaling.html")
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"ERROR: input Markdown not found: {args.input}")

    md = args.input.read_text(encoding="utf-8")
    body = md_to_body(md)
    args.output.write_text(wrap_html(body), encoding="utf-8")

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
