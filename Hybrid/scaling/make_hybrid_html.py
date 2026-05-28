#!/usr/bin/env python3
from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MD = ROOT / "hybrid_scaling.md"
OUT = ROOT / "hybrid_scaling.html"


def inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"`([^`]*)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    return s


def table(lines: list[str]) -> str:
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    head, body = rows[0], rows[2:]
    out = ['<div class="table-wrap"><table><thead><tr>']
    out += [f"<th>{inline(h)}</th>" for h in head]
    out += ["</tr></thead><tbody>"]
    for r in body:
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
    out += ["</tbody></table></div>"]
    return "\n".join(out)


def main() -> None:
    lines = MD.read_text().splitlines()
    body = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("# "):
            body.append(f"<h1>{inline(s[2:])}</h1>")
        elif s.startswith("## "):
            body.append(f"<h2>{inline(s[3:])}</h2>")
        elif s.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            body.append(table(block))
            continue
        else:
            body.append(f"<p>{inline(s)}</p>")
        i += 1

    OUT.write_text("""<!doctype html>
<html><head><meta charset="utf-8"><title>Hybrid MPI+CUDA scaling</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;margin:32px;background:#f6f8fb;color:#1f2937}main{max-width:1200px;margin:auto}h1{font-size:2rem}h2{border-bottom:1px solid #ddd;padding-bottom:6px;margin-top:30px}.table-wrap{overflow-x:auto;background:white;border:1px solid #ddd;border-radius:12px;margin:16px 0}table{border-collapse:collapse;width:100%;min-width:900px}th,td{padding:9px 11px;border-bottom:1px solid #e5e7eb;text-align:right;white-space:nowrap}th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){text-align:left}code{background:#eee;border-radius:4px;padding:1px 5px}
</style></head><body><main>
""" + "\n".join(body) + "\n</main></body></html>\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
