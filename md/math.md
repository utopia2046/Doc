# Math formulas in Markdown and in browsers

## Extensions in Visual Studio Code

 - [x] Markdown All in One
     https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one

 - [ ] Markdown+Math
    https://marketplace.visualstudio.com/items?itemName=goessner.mdmath

### Basic formula syntax

Anything in starts and ends with `$` are rendered as formula:

$x + y = z$

To place a formula in center, use double `$$`:

$$ x + y = z $$

If there is additional space after `$`, it will NOT be rendered as formula.

$ x + y = z $

To show `$`, could use `\$`

\$escape\$

In Visual Studio Code, type `$\`, all supported symbols are listed on UI. Following is a table of common symbols.

Category|Symbol|Script
-|-|-
Greek character|\alpha|$\alpha$
Greek character|\beta|$\beta$
Greek character|\gamma|$\gamma$
Greek character|\delta|$\delta$
Greek character|\epsilon|$\epsilon$
Greek character|\zeta|$\zeta$
Greek character|\eta|$\eta$
Greek character|\theta|$\theta$
Greek character|\iota|$\iota$
Greek character|\kappa|$\kappa$
Greek character|\lambda|$\lambda$
Greek character|\mu|$\mu$
Greek character|\nu|$\nu$
Greek character|\xi|$\xi$
Greek character|\omicron|$\omicron$
Greek character|\pi|$\pi$
Greek character|\rho|$\rho$
Greek character|\sigma|$\sigma$
Greek character|\tau|$\tau$
Greek character|\upsilon|$\upsilon$
Greek character|\phi|$\phi$
Greek character|\chi|$\chi$
Greek character|\psi|$\psi$
Greek character|\omega|$\omega$
Accents|a'|$a'$
Accents|\tilde{a}|$\tilde{a}$
Accents|\bar{a}|$\bar{a}$
Accents|\hat{a}|$\hat{a}$
Delimeters|()|$()$
Delimeters|[]|$[]$
Delimeters|\{\}|$\{\}$
Delimeters|\vert{a}\vert|$\vert{a}\vert$
Delimeters|( \big( \Big( \bigg( \Bigg(|$( \big( \Big( \bigg( \Bigg($
Layout|a \atop b|$a \atop b$
Layout|\sqrt{\smash[b]{y}}\sqrt{y}|$\sqrt{\smash[b]{y}}\sqrt{y}$
Layout|{a}\space{a}\thinspace{a}\medspace{a}\thickspace{a}|${a}\space{a}\space{a}\thinspace{a}\medspace{a}\thickspace{a}$
Logic|\because|$\because$
Logic|\therefore|$\therefore$
Logic|\forall|$\forall$
Logic|\in|$\in$
Logic|\notin|$\notin$
Logic|\land|$\land$
Logic|\lor|$\lor$
Logic|\to|$\to$
Logic|\gets|$\gets$
Logic|\implies|$\implies$
Logic|\impliedby|$\impliedby$
Logic|\iff|$\iff$
Operator|\sum|$\sum$
Operator|\prod|$\prod$
Operator|\int|$\int$
Operator|\cdot|${a}\cdot{b}$
Operator|\times|$\times$
Operator|\div|$\div$
Operator|\sqrt{x}|$\sqrt{x}$
Operator|\sqrt[3]{x}|$\sqrt[3]{x}$
Relation|\approx|$\approx$
Relation|\lt|$\lt$
Relation|\gt|$\gt$
Relation|\le|$\le$
Relation|\ge|$\ge$
Fraction|\frac{a}{b}|$\frac{a}{b}$
Fraction|\dfrac{a}{b}|$\dfrac{a}{b}$
Fraction|\cfrac{a}{1+\cfrac{1}{b}}|$\cfrac{a}{1+\cfrac{1}{b}}$
Arrow|\xrightarrow[under]{over}|$\xrightarrow[under]{over}$
Style|\displaystyle\sum_{i=1}^n|$\displaystyle\sum_{i=1}^n$
Style|\textstyle\sum_{i=1}^n|$\textstyle\sum_{i=1}^n$
Style|\lim\limits_x|$\lim\limits_x$

---
#### Matrix

`\begin{matrix} a & b \\ c & d \end{matrix}`

$\begin{matrix}
   a & b \\
   c & d
\end{matrix}$

`\begin{pmatrix} a & b \\ c & d \end{pmatrix}`

$\begin{pmatrix}
   a & b \\
   c & d
\end{pmatrix}$

`\begin{vmatrix} a & b \\ c & d \end{vmatrix}`

$\begin{vmatrix}
   a & b \\
   c & d
\end{vmatrix}$

`\begin{gathered} a=b \\ e=b+c \end{gathered}`

$\begin{gathered}
   a=b \\
   e=b+c
\end{gathered}$

`x = \begin{cases} a &\text{if } b \\ c &\text{if } d \end{cases}`

$x = \begin{cases}
   a &\text{if } b \\
   c &\text{if } d
\end{cases}$

#### Tag

`\tag{4a} x+y^{2x}`

$$\tag{4a} x+y^{2x}$$

#### Common

`\lim_{x \to \infty} \exp(-x) = 0`

$\lim_{x \to \infty} \exp(-x) = 0$

`k_{n+1} = n^2 + k_n^2 - k_{n-1}`

$k_{n+1} = n^2 + k_n^2 - k_{n-1}$

`f(n) = n^5 + 4n^2 + 2 |_{n=17}`

$f(n) = n^5 + 4n^2 + 2 |_{n=17}$

`\frac{n!}{k!(n-k)!} = \binom{n}{k}`

$\frac{n!}{k!(n-k)!} = \binom{n}{k}$

`\displaystyle\sum_{i=1}^{10} t_i`

$\displaystyle\sum_{i=1}^{10} t_i$

`\int_0^\infty \mathrm{e}^{-x}\,\mathrm{d}x`

$\int_0^\infty \mathrm{e}^{-x}\,\mathrm{d}x$

`Take $\frac{1}{2}$ cup of sugar, $\dots 3\times\frac{1}{2}=1\frac{1}{2}$`

Take $\frac{1}{2}$ cup of sugar, $\dots 3\times\frac{1}{2}=1\frac{1}{2}$

## KaTex syntax references

* https://khan.github.io/KaTeX/docs/supported.html
* https://en.wikibooks.org/wiki/LaTeX/Mathematics

## Show math formulas in browser

1. Reference KaTeX script(faster):
``` html
<script src="https://cdn.jsdelivr.net/npm/katex@0.10.0-beta/dist/katex.min.js" integrity="sha384-U8Vrjwb8fuHMt6ewaCy8uqeUXv4oitYACKdB0VziCerzt011iQ/0TqlSlv8MReCm" crossorigin="anonymous"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.0-beta/dist/katex.min.css" integrity="sha384-9tPv11A+glH/on/wEu99NVwDPwkMQESOocs/ZGXPoIiLE8MU/qkqUcZ3zzL+6DuH" crossorigin="anonymous">

In-browser rendering
<script>
katex.render("c = \\pm\\sqrt{a^2 + b^2}", element);
</script>

Server side rendering
<script>
var html = katex.renderToString("c = \\pm\\sqrt{a^2 + b^2}");
// generate string like '<span class="katex">...</span>'
</script>

Use auto-renderer
<script src="https://cdn.jsdelivr.net/npm/katex@0.10.0-beta/dist/contrib/auto-render.min.js" integrity="sha384-aGfk5kvhIq5x1x5YdvCp4upKZYnA8ckafviDpmWEKp4afOZEqOli7gqSnh8I6enH" crossorigin="anonymous"></script>

<body>
  ...
  <script>
    renderMathInElement(document.body);
  </script>
</body>
```

2. Reference mathjax script:
``` html
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML' async></script>

Configure mathjax
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>

<body>
When $a \ne 0$, there are two solutions to \(ax^2 + bx + c = 0\) and they are
$$x = {-b \pm \sqrt{b^2-4ac} \over 2a}.$$
</body>

```

References:
* https://khan.github.io/KaTeX/
* https://www.mathjax.org/#gettingstarted

