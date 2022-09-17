# -*- coding: utf-8-unix -*-

# EDITOR must wait!
EDITOR = nano
PREFIX = /usr/local

check:
	flake8 .
	flake8 . `grep -Fl '#!/usr/bin/env python3' bin/*`

clean:
	rm -rf *.lprof
	rm -rf *.prof
	rm -rf build
	rm -rf dataiter.egg-info
	rm -rf dist
	rm -rf doc/_build
	rm -rf doc/comparison/comparison.html
	rm -rf validation/*.csv
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf */*/__pycache__
	rm -rf .pytest_cache
	rm -rf */.pytest_cache
	rm -rf */*/.pytest_cache

doc:
	$(MAKE) -C doc clean html

doc-check:
	PYTHONPATH=. doc/check.py

doc-open:
	xdg-open doc/_build/html/index.html

# Watch files for changes, rebuild html.
# https://gist.github.com/senko/1154509
doc-watch:
	$(MAKE) -C doc clean html
	onchange $(MAKE) -C doc html

install:
	pip3 install .

# Non-essential scripts, not installed by default.
# Note that these don't go through setuptools rewriting,
# instead they just do a plain unspecified dataiter import.
install-cli:
	mkdir -p $(PREFIX)/bin
	for X in `ls bin | grep di-`; do \
	cp -fv bin/$$X $(PREFIX)/bin && \
	chmod +x $(PREFIX)/bin/$$X; \
	done

# Use @profile decorator from line-profiler.
# https://github.com/pyutils/line_profiler
profile:
	kernprof -lvu 1e-3 test.py

# Interactive!
publish:
	$(MAKE) check doc-check test validate clean
	python3 -m build
	test -s dist/dataiter-*-py3-none-any.whl
	test -s dist/dataiter-*.tar.gz
	ls -l dist
	@printf "Press Enter to upload or Ctrl+C to abort: "; read _
	twine upload dist/*
	sudo pip3 uninstall -y dataiter || true
	sudo pip3 uninstall -y dataiter || true
	sudo pip3 install -U dataiter
	$(MAKE) test-installed

# Interactive!
release:
	$(MAKE) check doc-check test validate clean
	@echo "BUMP VERSION NUMBERS"
	$(EDITOR) dataiter/__init__.py
	$(EDITOR) benchmark-versions.sh
	$(EDITOR) setup.py
	@echo "ADD RELEASE NOTES"
	$(EDITOR) NEWS.md
	sudo $(MAKE) install clean
	$(MAKE) test-installed
	tools/release

test:
	py.test .

test-installed:
	cd && python3 -c "import dataiter; dataiter.DataFrame()"
	cd && python3 -c "import dataiter; dataiter.ListOfDicts()"

validate:
	cd validation && DATAITER_USE_NUMBA=false ./validate-df.sh
	cd validation && DATAITER_USE_NUMBA=true  ./validate-df.sh
	cd validation && ./validate-ld.sh

.PHONY: check clean doc doc-check doc-open doc-watch install install-cli profile publish release test test-installed validate
