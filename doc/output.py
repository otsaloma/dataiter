# -*- coding: utf-8 -*-

import os
import subprocess
import sys

from pathlib import Path

CODE = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".")))
import dataiter as di
import json
import numpy as np
import traceback
from dataiter import dt
from dataiter import regex
di.PRINT_MAX_ITEMS = 3
di.PRINT_MAX_ROWS = 10
di.PRINT_MAX_WIDTH = 72
"""

def get_output(lines):
    try:
        return subprocess.check_output(
            # Unbuffered to keep stdout and stderr in order.
            args=[sys.executable, "-u", "-c", "\n".join(lines)],
            stderr=subprocess.STDOUT,
            cwd=Path("..").resolve(),
            # Free-threaded Python warns on importing modules that don't
            # declare support for running without the GIL, e.g. duckdb.
            env={**os.environ, "PYTHON_GIL": "1"},
            encoding="utf-8",
            errors="replace",
            timeout=30,
        ).splitlines()
    except subprocess.CalledProcessError as e:
        return e.output.splitlines()

def on_autodoc_process_docstring(app, what, name, obj, options, lines):
    print(f"Processing {name}...")
    # Intercept all ">>>" lines in docstring, run the corresponding code
    # in one process, separating the output of each line with a marker,
    # and inject any possible output into the docstring.
    marker = "--- doc/output.py marker ---"
    code = CODE.strip().splitlines()
    positions = []
    for i, line in enumerate(lines):
        if not line.startswith(">>>"): continue
        line = line.lstrip("> ")
        if line.startswith("#"): continue
        # Some docstrings will, on purpose, have lines of code that raise
        # errors. Wrap lines in try-except so that all lines will always be
        # executed. Show tracebacks only for lines that have output.
        if " = " in line or line.startswith(("from ", "import ")):
            code.append(f"try: {line}\nexcept Exception: pass")
            continue
        code.append(f"print({marker!r})")
        code.append(f"try:\n    print({line})\nexcept Exception:\n    traceback.print_exc()")
        positions.append(i + 1)
    if not positions: return
    blobs = [[]]
    for line in get_output(code):
        if line == marker:
            blobs.append([])
        else:
            blobs[-1].append(line)
    for i, blob in reversed(list(zip(positions, blobs[1:]))):
        for j in range(len(blob)):
            # Avoid a paragraph change on blank lines.
            if not blob[j].strip():
                blob[j] = "."
        lines[i:i] = blob

def setup(app):
    # https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-process-docstring
    app.connect("autodoc-process-docstring", on_autodoc_process_docstring)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
