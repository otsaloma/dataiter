#!/usr/bin/env python3

import re

from pathlib import Path

lines = []
text = Path("index.html").read_text("utf-8").strip()
print("Compiling index.html + blocks → comparison.html...")
print(f"index.html: {len(text)}")
for line in text.splitlines():
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
print(f"comparison.html: {len(text)}")
Path("comparison.html").write_text(text, "utf-8")

text = Path("prism.css").read_text("utf-8").strip()
if "font-family:" in text or "font-size" in text:
    # Strip Prism font rules so that they don't override
    # Tailwind CSS's better-thought-out default system font stack.
    # https://tailwindcss.com/docs/font-family
    text_length_prior = len(text)
    print("Patching prims.css... ", end="")
    text = re.sub(r"font-family:.+?;", "", text)
    text = re.sub(r"font-size:.+?;", "", text)
    print(len(text) - text_length_prior)
    Path("prism.css").write_text(text + "\n", "utf-8")
