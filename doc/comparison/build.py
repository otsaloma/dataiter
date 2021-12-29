#!/usr/bin/env python3

from pathlib import Path

print("Compiling index.html + blocks → comparison.html...")

lines = []
index = Path("index.html").read_text("utf-8")
for line in index.splitlines():
    if not line.strip().startswith('<pre data-src="'):
        lines.append(line)
        continue
    path = Path(line.split('"')[1])
    assert path.exists()
    code = path.read_text("utf-8").strip()
    lang = path.suffix.lstrip(".").lower()
    html = f'<pre><code class="language-{lang}">{code}</code></pre>'
    for line in html.splitlines():
        lines.append(line)

text = "\n".join(lines) + "\n"
Path("comparison.html").write_text(text, "utf-8")
