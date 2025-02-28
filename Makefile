PROJECT=EyeTracking
AUTHOR=Pather Stevenson, Deise Santana Maia
PY3=python3
PYTEST=pytest
TEST_PATH=tests
SRC=./src
MAIN=src/main.py
export PYTHONPATH
SPHINXBUILD=sphinx-build
CONFIGPATH=.
SOURCEDOC=sourcedoc
DOC=doc
LIBS=requirements.txt

run:
	$(PY3) $(MAIN)

test:
	$(PYTEST) $(TEST_PATH) -vv -W ignore::DeprecationWarning

lib:
	pip install -r $(LIBS)

clean:
	rm -f *~ */*~
	rm -rf __pycache__ */__pycache__
	rm -rf .pytest_cache
	rm -rf $(DOC)
	rm -f $(PROJECT).zip

doc: author
	$(SPHINXBUILD) -c $(CONFIGPATH) -b html $(SOURCEDOC) $(DOC)

archive: clean
	zip -r $(PROJECT).zip .

author:
	sed -i -e 's/^project =.*/project = "Module $(PROJECT)"/g' conf.py
	sed -i -e 's/^copyright =.*/copyright = "2025, $(AUTHOR), CRIStAL - Univ. Lille"/g' conf.py
	sed -i -e 's/^author =.*/author = "$(AUTHOR)"/g' conf.py

.PHONY: clean doc archive author lib test run
