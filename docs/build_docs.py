#!/usr/bin/env python3
"""Build a single, self-contained HTML documentation site from the Markdown docs.

Zero dependencies (stdlib only), like the rest of TensorSketch's tooling. Parses `SUMMARY.md` for the
navigation, converts each referenced Markdown file to HTML with a small purpose-built converter
(headings, fenced code, tables, lists, blockquotes, inline formatting, cross-page links), and
emits `docs/site/index.html` — a production-looking docs site with a sidebar, an on-this-page
outline, offline syntax highlighting, copy buttons, search, and a light/dark toggle.

    uv run python docs/build_docs.py      # writes docs/site/index.html
"""

from __future__ import annotations

import base64
import html
import posixpath
import re
from pathlib import Path

DOCS = Path(__file__).parent
OUT = DOCS / "site" / "index.html"

# ---------------------------------------------------------------------------- helpers


def slug(rel_path: str) -> str:
    """A DOM-safe page id from a docs-relative path, e.g. concepts/nodes.md -> concepts-nodes."""
    stem = rel_path[:-3] if rel_path.endswith(".md") else rel_path
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")


def heading_id(page: str, text: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return f"{page}--{base}"


def resolve_link(href: str, cur_dir: str, pages: set[str]) -> str:
    """Map a Markdown link to an in-site anchor when it targets another doc page."""
    raw = href.split("#", 1)[0]
    if not raw.endswith(".md"):
        return href  # external URL or a source-file link — leave it be
    target = posixpath.normpath(posixpath.join(cur_dir, raw))
    return f"#{slug(target)}" if target in pages else href


# ---------------------------------------------------------------------------- inline


def inline(text: str, cur_dir: str, pages: set[str]) -> str:
    """Convert inline Markdown (code, bold, italic, links, images) within one line of text."""
    codes: list[str] = []

    def stash_code(m: re.Match[str]) -> str:
        codes.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash_code, text)
    text = html.escape(text, quote=False)

    def image(m: re.Match[str]) -> str:
        alt, src = m.group(1), m.group(2)
        path = (DOCS / posixpath.normpath(posixpath.join(cur_dir, src))).resolve()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".svg"}:
            mime = "image/svg+xml" if path.suffix == ".svg" else f"image/{path.suffix[1:]}"
            data = base64.b64encode(path.read_bytes()).decode()
            return f'<figure><img alt="{html.escape(alt)}" src="data:{mime};base64,{data}"></figure>'
        return f'<figure class="img-todo"><span>🖼 {html.escape(alt)}</span></figure>'

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image, text)

    def link(m: re.Match[str]) -> str:
        label, href = m.group(1), m.group(2)
        dst = resolve_link(href, cur_dir, pages)
        ext = "" if dst.startswith("#") else ' target="_blank" rel="noopener"'
        return f'<a href="{html.escape(dst)}"{ext}>{label}</a>'

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"(?<![\w_])_([^_\n]+)_(?![\w_])", r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], text)


# ---------------------------------------------------------------------------- blocks


def convert(md: str, page: str, cur_dir: str, pages: set[str]) -> tuple[str, str]:
    """Convert a Markdown document to HTML; also return the page title (first H1)."""
    lines = md.splitlines()
    out: list[str] = []
    title = page
    i = 0
    n = len(lines)

    def flush_para(buf: list[str]) -> None:
        if buf:
            out.append(f"<p>{inline(' '.join(buf), cur_dir, pages)}</p>")
            buf.clear()

    para: list[str] = []
    while i < n:
        line = lines[i]

        if line.startswith("```"):  # fenced code
            flush_para(para)
            lang = line[3:].strip() or "text"
            i += 1
            code: list[str] = []
            while i < n and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            body = html.escape("\n".join(code))
            out.append(
                f'<div class="code" data-lang="{lang}"><button class="copy">Copy</button>'
                f'<pre><code class="lang-{lang}">{body}</code></pre></div>'
            )
            continue

        if re.match(r"^#{1,6}\s", line):  # heading
            flush_para(para)
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            if level == 1:
                title = text
                out.append(f"<h1>{inline(text, cur_dir, pages)}</h1>")
            else:
                hid = heading_id(page, text)
                out.append(f'<h{level} id="{hid}">{inline(text, cur_dir, pages)}</h{level}>')
            i += 1
            continue

        if re.match(r"^\s*\|.*\|\s*$", line) and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            flush_para(para)
            i = _table(lines, i, out, cur_dir, pages)
            continue

        if line.startswith(">"):  # blockquote
            flush_para(para)
            quote: list[str] = []
            while i < n and lines[i].startswith(">"):
                quote.append(lines[i].lstrip(">").strip())
                i += 1
            out.append(f"<blockquote>{inline(' '.join(quote), cur_dir, pages)}</blockquote>")
            continue

        if re.match(r"^\s*([-*]|\d+\.)\s+", line):  # list (possibly nested)
            flush_para(para)
            i = _list(lines, i, out, cur_dir, pages)
            continue

        if re.match(r"^(-{3,}|\*{3,}|_{3,})\s*$", line):  # horizontal rule
            flush_para(para)
            out.append("<hr>")
            i += 1
            continue

        if not line.strip():  # blank
            flush_para(para)
            i += 1
            continue

        if line.lstrip().startswith("<!--"):  # skip HTML comments (diagram TODOs etc.)
            i += 1
            continue

        para.append(line.strip())
        i += 1

    flush_para(para)
    return "\n".join(out), title


def _table(lines: list[str], i: int, out: list[str], cur_dir: str, pages: set[str]) -> int:
    def cells(row: str) -> list[str]:
        return [c.strip() for c in row.strip().strip("|").split("|")]

    header = cells(lines[i])
    i += 2  # header + separator
    rows: list[list[str]] = []
    while i < len(lines) and re.match(r"^\s*\|.*\|\s*$", lines[i]):
        rows.append(cells(lines[i]))
        i += 1
    head = "".join(f"<th>{inline(c, cur_dir, pages)}</th>" for c in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{inline(c, cur_dir, pages)}</td>" for c in r) + "</tr>" for r in rows
    )
    out.append(f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")
    return i


def _list(lines: list[str], i: int, out: list[str], cur_dir: str, pages: set[str]) -> int:
    items: list[tuple[int, bool, str]] = []
    while i < len(lines):
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
        if not m:
            if lines[i].strip() == "":  # allow a blank line to end the list
                break
            # a wrapped continuation line — append to the last item
            if items and lines[i].startswith(" "):
                indent, ordered, txt = items[-1]
                items[-1] = (indent, ordered, txt + " " + lines[i].strip())
                i += 1
                continue
            break
        indent = len(m.group(1))
        ordered = m.group(2).endswith(".")
        items.append((indent, ordered, m.group(3)))
        i += 1

    # Render by indent depth using a simple stack of (indent, tag).
    html_parts: list[str] = []
    stack: list[tuple[int, str]] = []
    for indent, ordered, txt in items:
        tag = "ol" if ordered else "ul"
        while stack and indent < stack[-1][0]:
            html_parts.append(f"</{stack.pop()[1]}>")
        if not stack or indent > stack[-1][0]:
            stack.append((indent, tag))
            html_parts.append(f"<{tag}>")
        html_parts.append(f"<li>{inline(txt, cur_dir, pages)}</li>")
    while stack:
        html_parts.append(f"</{stack.pop()[1]}>")
    out.append("".join(html_parts))
    return i


# ---------------------------------------------------------------------------- nav


def parse_summary() -> list[dict[str, object]]:
    """Parse SUMMARY.md into an ordered nav: section headers and page links."""
    nav: list[dict[str, object]] = []
    for line in (DOCS / "SUMMARY.md").read_text().splitlines():
        sec = re.match(r"^\s*-\s+\*\*(.+?)\*\*\s*$", line)
        page = re.match(r"^\s*-\s+\[(.+?)\]\(([^)]+\.md)\)\s*$", line)
        if sec:
            nav.append({"type": "section", "title": sec.group(1)})
        elif page:
            nav.append({"type": "page", "title": page.group(1), "path": page.group(2)})
    return nav


# ---------------------------------------------------------------------------- build


def main() -> None:
    nav = parse_summary()
    page_paths = [n["path"] for n in nav if n["type"] == "page"]  # type: ignore[index]
    pages = set(page_paths)  # type: ignore[arg-type]

    articles: list[str] = []
    nav_html: list[str] = []
    first = True
    for item in nav:
        if item["type"] == "section":
            nav_html.append(f'<div class="nav-section">{html.escape(str(item["title"]))}</div>')
            continue
        rel = str(item["path"])
        pslug = slug(rel)
        cur_dir = posixpath.dirname(rel)
        body, title = convert((DOCS / rel).read_text(), pslug, cur_dir, pages)
        hidden = "" if first else " hidden"
        articles.append(f'<article class="page" id="{pslug}"{hidden}>{body}</article>')
        nav_html.append(
            f'<a class="nav-link" href="#{pslug}" data-page="{pslug}">{html.escape(str(item["title"]))}</a>'
        )
        first = False

    out_html = (
        TEMPLATE.replace("__NAV__", "\n".join(nav_html))
        .replace("__PAGES__", "\n".join(articles))
        .replace("__CSS__", CSS)
        .replace("__JS__", JS)
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(out_html, encoding="utf-8")
    kb = len(out_html.encode()) / 1024
    print(f"wrote {OUT.relative_to(DOCS.parent)}  ({kb:.0f} KB, {len(page_paths)} pages)")


# ---------------------------------------------------------------------------- template

TEMPLATE = """<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TensorSketch — Documentation</title>
<style>__CSS__</style>
</head>
<body>
<header class="topbar">
  <button class="menu" id="menu" aria-label="Menu">☰</button>
  <a class="brand" href="#index"><span class="logo">✳</span> TensorSketch <span class="brand-sub">docs</span></a>
  <span class="pill">pre-1.0</span>
  <div class="spacer"></div>
  <input id="search" class="search" type="search" placeholder="Search pages…" autocomplete="off">
  <button class="theme" id="theme" aria-label="Toggle theme">◐</button>
</header>
<div class="layout">
  <nav class="sidebar" id="sidebar">__NAV__</nav>
  <main class="content" id="content">__PAGES__</main>
  <aside class="toc" id="toc"></aside>
</div>
<div class="scrim" id="scrim"></div>
<script>__JS__</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------- css

CSS = r"""
:root{
  --bg:#ffffff; --fg:#1f2328; --muted:#59636e; --faint:#818b98;
  --border:#d1d9e0; --border-soft:#e7ecf0; --sidebar:#f7f8fa; --hover:#eef1f4;
  --accent:#6741d9; --accent-soft:#f0ebfc; --accent-ink:#5a34c9;
  --code-bg:#0d1117; --code-fg:#e6edf3; --sh:0 1px 2px rgba(31,35,40,.06);
  --k:#ff7b72; --s:#a5d6ff; --c:#8b949e; --n:#79c0ff; --b:#d2a8ff; --f:#d2a8ff; --p:#7ee787;
}
html[data-theme=dark]{
  --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --faint:#7d8590;
  --border:#30363d; --border-soft:#21262d; --sidebar:#0b0e13; --hover:#161b22;
  --accent:#a371f7; --accent-soft:#1c1630; --accent-ink:#b587f7; --sh:none;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
a{color:var(--accent-ink);text-decoration:none}
a:hover{text-decoration:underline}

.topbar{position:sticky;top:0;z-index:30;display:flex;align-items:center;gap:12px;
  height:56px;padding:0 18px;background:var(--bg);border-bottom:1px solid var(--border)}
.brand{display:flex;align-items:center;gap:8px;font-weight:700;font-size:17px;color:var(--fg)}
.brand:hover{text-decoration:none}
.brand .logo{color:var(--accent)}
.brand-sub{color:var(--faint);font-weight:500}
.pill{font-size:11px;font-weight:600;color:var(--accent-ink);background:var(--accent-soft);
  border:1px solid var(--border-soft);padding:2px 8px;border-radius:999px}
.spacer{flex:1}
.search{width:210px;max-width:38vw;height:34px;padding:0 12px;border:1px solid var(--border);
  border-radius:8px;background:var(--bg);color:var(--fg);font-size:13px}
.search:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
.theme,.menu{height:34px;min-width:34px;border:1px solid var(--border);border-radius:8px;
  background:var(--bg);color:var(--fg);cursor:pointer;font-size:15px}
.menu{display:none}

.layout{display:grid;grid-template-columns:270px minmax(0,1fr) 220px;
  max-width:1400px;margin:0 auto;align-items:start}
.sidebar{position:sticky;top:56px;height:calc(100vh - 56px);overflow-y:auto;
  padding:22px 14px 60px;background:var(--sidebar);border-right:1px solid var(--border)}
.nav-section{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  color:var(--faint);margin:20px 10px 6px}
.nav-section:first-child{margin-top:0}
.nav-link{display:block;padding:6px 10px;border-radius:7px;color:var(--muted);
  font-size:13.5px;border-left:2px solid transparent}
.nav-link:hover{background:var(--hover);color:var(--fg);text-decoration:none}
.nav-link.active{background:var(--accent-soft);color:var(--accent-ink);font-weight:600;
  border-left-color:var(--accent)}
.nav-link.hidden{display:none}

.content{padding:38px 52px 120px;min-width:0}
.page{display:none;max-width:760px}
.page.active{display:block;animation:fade .18s ease}
@keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}

h1{font-size:32px;line-height:1.2;letter-spacing:-.02em;margin:0 0 18px;font-weight:750}
h2{font-size:22px;margin:38px 0 12px;padding-top:14px;border-top:1px solid var(--border-soft);
  letter-spacing:-.01em;font-weight:700}
h3{font-size:17px;margin:26px 0 8px;font-weight:650}
h4{font-size:15px;margin:20px 0 6px;color:var(--muted)}
p{margin:0 0 14px}
strong{font-weight:650}
hr{border:none;border-top:1px solid var(--border-soft);margin:30px 0}
ul,ol{margin:0 0 14px;padding-left:22px}
li{margin:5px 0}
li>ul,li>ol{margin:5px 0}

:not(pre)>code{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  font-size:.86em;background:var(--accent-soft);color:var(--accent-ink);
  padding:.12em .4em;border-radius:5px;border:1px solid var(--border-soft)}
html[data-theme=dark] :not(pre)>code{background:#1a2230;color:#c7d2fe;border-color:#30363d}

.code{position:relative;margin:0 0 18px}
.code::before{content:attr(data-lang);position:absolute;top:8px;right:12px;font-size:10.5px;
  letter-spacing:.08em;text-transform:uppercase;color:#7d8590;font-weight:600}
.code .copy{position:absolute;top:8px;right:58px;font-size:11px;color:#adbac7;background:#1c2430;
  border:1px solid #30363d;border-radius:6px;padding:3px 9px;cursor:pointer;opacity:0;transition:.15s}
.code:hover .copy{opacity:1}
.code .copy:hover{background:#26303c;color:#fff}
.code .copy.done{color:#7ee787;border-color:#238636}
pre{margin:0;background:var(--code-bg);border:1px solid #1f2733;border-radius:11px;
  padding:16px 18px;overflow:auto}
pre code{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  font-size:13px;line-height:1.6;color:var(--code-fg)}
.tok-k{color:var(--k)}.tok-s{color:var(--s)}.tok-c{color:var(--c);font-style:italic}
.tok-n{color:var(--n)}.tok-b{color:var(--b)}.tok-f{color:var(--f)}.tok-d{color:var(--b)}
.tok-o{color:#ff7b72}.tok-cons{color:var(--n)}.tok-p{color:var(--p)}

blockquote{margin:0 0 18px;padding:12px 16px;background:var(--accent-soft);
  border:1px solid var(--border-soft);border-left:3px solid var(--accent);border-radius:0 8px 8px 0;
  color:var(--fg)}
blockquote p{margin:0}

.table-wrap{overflow-x:auto;margin:0 0 18px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{border:1px solid var(--border);padding:8px 12px;text-align:left;vertical-align:top}
th{background:var(--sidebar);font-weight:650}
tbody tr:nth-child(even){background:var(--border-soft)}

figure{margin:0 0 18px}
figure img{max-width:100%;border:1px solid var(--border);border-radius:10px;box-shadow:var(--sh)}
.img-todo{display:flex;align-items:center;justify-content:center;height:150px;color:var(--faint);
  background:var(--sidebar);border:1px dashed var(--border);border-radius:10px;font-size:13px}

.toc{position:sticky;top:56px;height:calc(100vh - 56px);overflow-y:auto;padding:38px 18px 60px}
.toc-title{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  color:var(--faint);margin-bottom:10px}
.toc a{display:block;padding:4px 0 4px 12px;color:var(--muted);font-size:12.5px;
  border-left:2px solid var(--border-soft);line-height:1.4}
.toc a.sub{padding-left:24px}
.toc a:hover{color:var(--fg);text-decoration:none}
.toc a.active{color:var(--accent-ink);border-left-color:var(--accent);font-weight:600}
.toc:empty::before{content:""}

.scrim{display:none;position:fixed;inset:56px 0 0;background:rgba(0,0,0,.35);z-index:20}

@media(max-width:1180px){.layout{grid-template-columns:250px minmax(0,1fr)}.toc{display:none}}
@media(max-width:820px){
  .layout{grid-template-columns:1fr}
  .menu{display:block}.search{width:150px}
  .sidebar{position:fixed;top:56px;left:0;width:280px;z-index:25;transform:translateX(-102%);
    transition:transform .2s ease}
  .sidebar.open{transform:none}
  .scrim.open{display:block}
  .content{padding:26px 22px 100px}
}
"""

# ---------------------------------------------------------------------------- js

JS = r"""
(function(){
  const KW = new Set(("def class return await async if elif else for while in not and or is with as "+
    "import from try except finally raise yield lambda pass break continue global nonlocal assert "+
    "del match case").split(" "));
  const CONST = new Set("True False None self cls __name__ NotImplemented Ellipsis".split(" "));
  const BUILT = new Set(("print len range list dict set tuple str int float bool sum sorted enumerate "+
    "zip isinstance issubclass super min max any all open getattr setattr hasattr type map filter "+
    "repr abs round next iter format dataclass field property staticmethod classmethod").split(" "));

  function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

  function hlPython(src){
    const re = /(?<c>#[^\n]*)|(?<s>[rbfRBF]{0,2}(?:\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\'|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))|(?<d>@[\w.]+)|(?<num>\b\d[\d_]*(?:\.\d+)?(?:e[-+]?\d+)?\b)|(?<w>[A-Za-z_]\w*)/g;
    let out='', last=0, m;
    while((m=re.exec(src))){
      out += esc(src.slice(last, m.index)); last = re.lastIndex;
      const g = m.groups;
      if(g.c) out += `<span class="tok-c">${esc(m[0])}</span>`;
      else if(g.s) out += `<span class="tok-s">${esc(m[0])}</span>`;
      else if(g.d) out += `<span class="tok-d">${esc(m[0])}</span>`;
      else if(g.num) out += `<span class="tok-n">${esc(m[0])}</span>`;
      else {
        const w=m[0];
        const cls = KW.has(w)?'tok-k':CONST.has(w)?'tok-cons':BUILT.has(w)?'tok-b':null;
        out += cls?`<span class="${cls}">${esc(w)}</span>`:esc(w);
      }
    }
    out += esc(src.slice(last));
    return out;
  }
  function hlBash(src){
    return esc(src).split("\n").map(l=>{
      if(/^\s*#/.test(l)) return `<span class="tok-c">${l}</span>`;
      l = l.replace(/(^|\s)(--?[\w-]+)/g,'$1<span class="tok-f">$2</span>');
      l = l.replace(/(&quot;[^&]*&quot;|&#39;[^&]*&#39;)/g,'<span class="tok-s">$1</span>');
      l = l.replace(/^(\s*)(uv|pip|python|make|cd|git|npx|export|ruff|mypy|pytest)\b/,'$1<span class="tok-k">$2</span>');
      return l;
    }).join("\n");
  }
  function hlJson(src){
    let s = esc(src);
    s = s.replace(/(&quot;(?:\\.|[^&])*?&quot;)(\s*:)/g,'<span class="tok-b">$1</span>$2');
    s = s.replace(/(:\s*)(&quot;(?:\\.|[^&])*?&quot;)/g,'$1<span class="tok-s">$2</span>');
    s = s.replace(/\b(true|false|null)\b/g,'<span class="tok-cons">$1</span>');
    s = s.replace(/(:\s*)(-?\d[\d.eE+-]*)/g,'$1<span class="tok-n">$2</span>');
    return s;
  }
  function highlight(){
    document.querySelectorAll('pre code').forEach(el=>{
      const lang = (el.className.match(/lang-(\w+)/)||[])[1]||'text';
      const raw = el.textContent;
      if(lang==='python'||lang==='py') el.innerHTML = hlPython(raw);
      else if(lang==='bash'||lang==='sh'||lang==='shell') el.innerHTML = hlBash(raw);
      else if(lang==='json') el.innerHTML = hlJson(raw);
    });
  }

  function buildTOC(page){
    const toc = document.getElementById('toc');
    const heads = page.querySelectorAll('h2, h3');
    if(!heads.length){ toc.innerHTML=''; return; }
    let h = '<div class="toc-title">On this page</div>';
    heads.forEach(el=>{
      const sub = el.tagName==='H3' ? ' sub' : '';
      h += `<a class="${sub.trim()}" href="#${el.id}" data-tid="${el.id}">${el.textContent}</a>`;
    });
    toc.innerHTML = h;
    toc.querySelectorAll('a').forEach(a=>a.addEventListener('click',e=>{
      e.preventDefault();
      document.getElementById(a.dataset.tid).scrollIntoView({behavior:'smooth',block:'start'});
    }));
  }

  function route(){
    const id = (location.hash.replace('#','').split('--')[0]) || 'index';
    const page = document.getElementById(id) || document.querySelector('.page');
    document.querySelectorAll('.page').forEach(p=>{p.classList.remove('active');p.hidden=true;});
    page.hidden=false; page.classList.add('active');
    document.querySelectorAll('.nav-link').forEach(a=>
      a.classList.toggle('active', a.dataset.page===page.id));
    buildTOC(page);
    const anchor = location.hash.includes('--') ? document.getElementById(location.hash.slice(1)) : null;
    (anchor||document.getElementById('content')).scrollIntoView({block: anchor?'start':'start'});
    if(!anchor) window.scrollTo(0,0);
    closeNav();
  }

  // sidebar (mobile) + scrim
  const sb=document.getElementById('sidebar'), scrim=document.getElementById('scrim');
  function closeNav(){sb.classList.remove('open');scrim.classList.remove('open');}
  document.getElementById('menu').addEventListener('click',()=>{sb.classList.toggle('open');scrim.classList.toggle('open');});
  scrim.addEventListener('click',closeNav);

  // search filters the sidebar; Enter jumps to the first match
  const search=document.getElementById('search');
  search.addEventListener('input',()=>{
    const q=search.value.trim().toLowerCase();
    document.querySelectorAll('.nav-link').forEach(a=>
      a.classList.toggle('hidden', q && !a.textContent.toLowerCase().includes(q)));
    document.querySelectorAll('.nav-section').forEach(s=>s.style.opacity = q?.4:1);
  });
  search.addEventListener('keydown',e=>{
    if(e.key==='Enter'){const first=[...document.querySelectorAll('.nav-link:not(.hidden)')][0];
      if(first){location.hash=first.getAttribute('href');search.value='';search.dispatchEvent(new Event('input'));}}
  });

  // theme
  const themeBtn=document.getElementById('theme');
  const saved=localStorage.getItem('tensorsketch-docs-theme'); if(saved) document.documentElement.dataset.theme=saved;
  themeBtn.addEventListener('click',()=>{
    const t=document.documentElement.dataset.theme==='dark'?'light':'dark';
    document.documentElement.dataset.theme=t; localStorage.setItem('tensorsketch-docs-theme',t);
  });

  // copy buttons
  document.querySelectorAll('.code').forEach(box=>{
    const btn=box.querySelector('.copy'), code=box.querySelector('code');
    btn.addEventListener('click',()=>{navigator.clipboard.writeText(code.textContent).then(()=>{
      btn.textContent='Copied'; btn.classList.add('done');
      setTimeout(()=>{btn.textContent='Copy';btn.classList.remove('done');},1400);});});
  });

  // active TOC entry on scroll
  const spy=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){
    const a=document.querySelector(`.toc a[data-tid="${e.target.id}"]`);
    if(a){document.querySelectorAll('.toc a').forEach(x=>x.classList.remove('active'));a.classList.add('active');}
  }});},{rootMargin:'-56px 0px -70% 0px'});
  function observeHeads(){spy.disconnect();document.querySelectorAll('.page:not([hidden]) h2,.page:not([hidden]) h3').forEach(h=>spy.observe(h));}

  highlight();
  window.addEventListener('hashchange',()=>{route();observeHeads();});
  route(); observeHeads();
})();
"""


if __name__ == "__main__":
    main()
