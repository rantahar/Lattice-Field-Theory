

#html: Makefile build/lattice.tex src/lattice.lyx
#	pandoc --toc  --toc-depth=2 --mathjax --template src/template.html -f latex -t html < build/lattice.tex > public/index.html

html: Makefile src/lattice.md
	pandoc --toc --toc-depth=2 --mathjax --template src/template.html -f markdown -t html -i src/lattice.md -o public/index.html


#pdf: Makefile src/lattice.lyx
#	lyx -e pdf src/lattice.lyx
#	mv src/lattice.pdf lattice.pdf

#build/lattice.tex: Makefile src/lattice.lyx
#	mkdir -p build
#	lyx -e latex src/lattice.lyx
#	mv src/lattice.tex build/lattice.tex

clean:
	rm -fr build