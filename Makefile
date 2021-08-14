.PHONY: init clean lint test
.DEFAULT_GOAL := build

init:
	pip install -r requirements.txt

lint:
	python -m flake8 ./lassobbn

test: clean lint
	nosetests tests --exe

build: test
	python setup.py bdist_egg

build-dist:
	python setup.py bdist_egg sdist bdist_wheel

install: build
	python setup.py install

compile:
	python -m compileall -f ./lassobbn

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	rm -fr coverage/
	rm -fr dist/
	rm -fr build/
	rm -fr lassobbn.egg-info/
	rm -fr jupyter/.ipynb_checkpoints/
	rm -fr joblib_memmap/
	rm -fr docs/build/
	rm -fr .pytest_cache/
	rm -f .coverage
	rm -f .noseids

