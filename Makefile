EMACS = emacsclient
README = "README"

export: pdf md html

pdf:
	$(EMACS) --eval '(progn (find-file $(README)) (org-latex-export-to-pdf))'
md:
	$(EMACS) --eval '(progn (find-file $(README)) (org-md-export-to-markdown))'
html:
	$(EMACS) --eval '(progn (find-file $(README)) (org-html-export-to-html))'
tangle:
	$(EMACS) --eval '(progn (find-file $(README)) (org-babel-tangle))'
build:
	poetry build

depends:
	poetry install -E doc -E dev -E test

tox:
	poetry run tox

test:
	poetry run pytest

commit:
	poetry run pre-commit

docs: docs-deploy docs-serve

docs-deploy:
	poetry run mike set-default `poetry version --short`
	poetry run mike deploy -p `poetry version --short`
docs-serve:
	poetry run mike serve

docs-build:
	poetry run mike build
