# -*- coding: utf-8 -*-

import os
import subprocess

CODE = """
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import dataiter as di
di.PRINT_MAX_WIDTH = 64
"""

def get_output(lines):
    return subprocess.check_output(
        args=["python3", "-c", ";".join(lines)],
        cwd=os.path.abspath(".."),
        encoding="utf_8",
        errors="replace",
        universal_newlines=True,
        text=True,
        timeout=30,
    ).splitlines()

def on_autodoc_process_docstring(app, what, name, obj, options, lines):
    # Intercept all ">>>" lines in docstring, run the corresponding code
    # and inject any possible output into the docstring.
    code = CODE.strip().splitlines()
    output = []
    for i, line in enumerate(lines):
        if not line.startswith(">>>"): continue
        line = line.lstrip("> ")
        code.append(line)
        if " = " in line: continue
        blob = get_output(code[:-1] + [f"print({code[-1]})"])
        for j in range(len(blob)):
            # Avoid a paragraph change on blank lines.
            if not blob[j].strip():
                blob[j] = "."
        output.append((i + 1, blob))
    for i, blob in reversed(output):
        lines[i:i] = blob

def setup(app):
    # https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-process-docstring
    app.connect("autodoc-process-docstring", on_autodoc_process_docstring)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }