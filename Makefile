

html: Makefile src/lattice.md
	pandoc --toc --toc-depth=2 --mathjax --template src/html-template.html -f markdown -t html -i src/lattice.md -o public/index.html


pdf: Makefile src/lattice.md
	mkdir -p public
	pandoc --toc --toc-depth=2 --mathjax --template src/pdf-template.html -f markdown -t html -i src/lattice.md -o lattice.pdf

html-lyx: Makefile build/lattice.tex src/lattice.lyx
	pandoc --toc --toc-depth=2 --mathjax --template src/html-template.html -f markdown -t html -i src/lattice.md -o lattice.pdf

pdf-lyx: Makefile src/lattice.lyx
	lyx -e pdf src/lattice.lyx
	mv src/lattice.pdf lattice.pdf

build/lattice.tex: Makefile src/lattice.lyx
	mkdir -p build
	lyx -e latex src/lattice.lyx
	mv src/lattice.tex build/lattice.tex

clean:
	rm -fr build

