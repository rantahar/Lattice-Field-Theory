
lyx -e latex lattice.lyx
pandoc -s --mathjax="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" -f latex -t html < src/lattice.tex > index.html

