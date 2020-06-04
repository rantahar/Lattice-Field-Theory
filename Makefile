

html: Makefile build/lattice.tex src/lattice.lyx
	mkdir -p public
	pandoc -s --toc --mathjax="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" --css bootstrap-3.3.7.min.css -f latex -t html < build/lattice.tex > public/index.html

pdf: Makefile src/lattice.lyx
	lyx -e pdf src/lattice.lyx
	mv src/lattice.pdf lattice.pdf

build/lattice.tex: Makefile src/lattice.lyx
	mkdir -p build
	lyx -e latex src/lattice.lyx
	mv src/lattice.tex build/lattice.tex

clean:
	rm -fr build