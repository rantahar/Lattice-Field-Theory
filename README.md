
lyx -e latex lattice.lyx
pandoc -f latex -t html < lattice.tex > index.html
