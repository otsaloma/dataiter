# -*- coding: utf-8 -*-

import os
import subprocess

CODE = """
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import dataiter as di
import numpy as np
di.PRINT_MAX_ITEMS = 3
di.PRINT_MAX_ROWS = 10
di.PRINT_MAX_WIDTH = 72
"""

def get_output(lines):
    try:
        return subprocess.check_output(
            args=["python3", "-c", "\n".join(lines)],
            stderr=subprocess.STDOUT,
            cwd=os.path.abspath(".."),
            encoding="utf_8",
            errors="replace",
            universal_newlines=True,
            text=True,
            timeout=30,
        ).splitlines()
    except subprocess.CalledProcessError as e:
        return e.output.splitlines()

def on_autodoc_process_docstring(app, what, name, obj, options, lines):
    print(f"Processing {name}...")
    # Intercept all ">>>" lines in docstring, run the corresponding code
    # and inject any possible output into the docstring.
    code = CODE.strip().splitlines()
    output = []
    for i, line in enumerate(lines):
        if not line.startswith(">>>"): continue
        line = line.lstrip("> ")
        if line.startswith("#"): continue
        # Some docstrings will, on purpose, have lines of code that raise
        # errors. Wrap lines in try-except so that all lines will always be
        # executed and output from only the last line will be used.
        code.append(f"try: {line}\nexcept Exception: pass")
        if " = " in line: continue
        if line.startswith(("from ", "import ")): continue
        blob = get_output(code[:-1] + [f"print({line})"])
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
