all: black pytest

black: 
	black --check `find src tests scripts -iname "*.py"`

pytest:
	pytest


.PHONY: all
