# -*- coding: utf-8-unix -*-

PREFIX = /usr/local

check:
	flake8 .

clean:
	rm -rf build
	rm -rf dataiter.egg-info
	rm -rf dist
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf .pytest_cache
	rm -rf */.pytest_cache

install:
	./setup.py install --prefix=$(PREFIX)

push:
	$(MAKE) clean
	./setup.py sdist bdist_wheel
	twine upload dist/*

test:
	py.test .

test-installed:
	cd && python3 -c "import dataiter; dataiter.ListOfDicts([])"

.PHONY: check clean install push test test-installed
