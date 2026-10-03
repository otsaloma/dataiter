# -*- coding: utf-8-unix -*-

# EDITOR must wait!
EDITOR = nano
PREFIX = /usr/local

# Latest Numba-compatible
PYTHON = python3.14

check:
	flake8 .
	flake8 . `grep -Fl '#!/usr/bin/env python3' bin/*`

clean:
	rm -rf build
	rm -rf dist
	rm -rf doc/_build
	rm -rf doc/comparison/comparison.html
	rm -rf validation/*.csv
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +

doc:
	$(MAKE) SPHINXBUILD=../venv/bin/sphinx-build -C doc clean html

doc-check:
	PYTHONPATH=. doc/check.py

doc-open:
	xdg-open doc/_build/html/index.html

doc-watch:
	watchexec -e py,rst --workdir doc $(MAKE) SPHINXBUILD=../venv/bin/sphinx-build html

install:
	pip install --break-system-packages .

# Non-essential scripts, not installed by default.
# Note that these don't go through setuptools rewriting,
# instead they just do a plain unspecified dataiter import.
install-cli:
	mkdir -p $(PREFIX)/bin
	for X in `ls bin | grep di-`; do \
	cp -fv bin/$$X $(PREFIX)/bin && \
	chmod +x $(PREFIX)/bin/$$X; \
	done

# Interactive!
publish:
	$(MAKE) clean
	python -m build
	test -s dist/dataiter-*-py3-none-any.whl
	test -s dist/dataiter-*.tar.gz
	twine check dist/*
	ls -l dist
	@printf "Press Enter to upload or Ctrl+C to abort: "; read _
	twine upload dist/*

# Interactive!
release:
	$(MAKE) check doc-check test validate clean
	@echo "BUMP VERSION NUMBERS"
	$(EDITOR) bin/di-open
	$(EDITOR) dataiter/__init__.py
	$(EDITOR) benchmark-versions.sh
	@echo "ADD RELEASE NOTES"
	$(EDITOR) NEWS.md
	sudo $(MAKE) install clean
	$(MAKE) test-installed
	tools/release

test:
	pytest .

test-installed:
	cd && python -c "import dataiter; dataiter.DataFrame()"
	cd && python -c "import dataiter; dataiter.ListOfDicts()"

validate:
	cd validation && DATAITER_USE_NUMBA=false ./validate-df.sh
	cd validation && DATAITER_USE_NUMBA=true ./validate-df.sh
	cd validation && ./validate-ld.sh

venv:
	rm -rf venv
	$(PYTHON) -m venv venv
	. venv/bin/activate && \
	  pip install -U pip setuptools wheel && \
	  pip install -r requirements.txt

.PHONY: check clean doc doc-check doc-open doc-watch install install-cli publish release test test-installed validate venv
