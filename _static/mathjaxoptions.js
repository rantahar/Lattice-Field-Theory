
window.MathJax = {
  loader: {
    load: ['[tex]/boldsymbol'],
  },
  tex: {
    tags: 'ams',
    packages: {
      '[+]': ['boldsymbol'],
      '[+]': ['dsfont'],
      '[+]': ['amsmath'],
      '[+]': ['mathtools']},
      macros: {  mathlarger: ["{\\large \#1}",1], dsone: "\\unicode{x1D7D9}" }
  },
};
