all: black pytest

black: 
	black --check `find . -iname "*.py"`

pytest:
	pytest


.PHONY: all
