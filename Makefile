# -*- coding: utf-8-unix -*-

# EDITOR must wait!
EDITOR = nano
PREFIX = /usr/local

check:
	flake8 .
	flake8 bin/*

clean:
	rm -rf build
	rm -rf dataiter.egg-info
	rm -rf dist
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf */*/__pycache__
	rm -rf .pytest_cache
	rm -rf */.pytest_cache
	rm -rf */*/.pytest_cache

doc:
	$(MAKE) -C doc clean html

doc-open:
	xdg-open doc/_build/html/index.html

# Watch files for changes, rebuild html.
# https://gist.github.com/senko/1154509
doc-watch:
	$(MAKE) -C doc clean html
	onchange $(MAKE) -C doc html

install:
	./setup.py install --prefix=$(PREFIX)

# Interactive!
publish:
	$(MAKE) check test clean
	./setup.py sdist bdist_wheel
	test -s dist/dataiter-*-py3-none-any.whl
	test -s dist/dataiter-*.tar.gz
	ls -l dist
	@printf "Press Enter to upload or Ctrl+C to abort: "; read _
	twine upload dist/*
	sudo pip3 uninstall -y dataiter || true
	sudo pip3 uninstall -y dataiter || true
	sudo pip3 install dataiter
	$(MAKE) test-installed

# Interactive!
release:
	$(MAKE) check test clean
	@echo "BUMP VERSION NUMBERS"
	$(EDITOR) dataiter/__init__.py
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

.PHONY: check clean doc doc-open doc-watch install publish release test test-installed
