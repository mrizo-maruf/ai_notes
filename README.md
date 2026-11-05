# AI Course Notes

LaTeX notes for **Fundamentals of Deep Learning** and **Machine Learning**.
Each chapter has its own file, and each section within it its own file, so notes can be added one at a time.

## Layout

```
common/
  preamble.tex          # packages, theorem envs, colored boxes, code style (shared)
  macros.tex            # math notation: \R, \E, \vec, \norm, \argmin, \loss, ...
  references.bib        # shared bibliography
  section_template.tex  # starting point for a new section
deep_learning/
  main.tex                          # document root: title, TOC, \include{chapters/NN_chapter}
  chapters/
    01_logreg_to_mlp.tex            # \chapter{...} + \input of its sections
    01_logreg_to_mlp/
      01_recap_of_logistic_regression.tex   # one file per \section
      ...
  figures/                          # images for this course (\includegraphics{name.png})
machine_learning/
  (same structure)
```

## Build

Needs `pdflatex`, `latexmk`, `bibtex` (TeX Live).

```bash
make            # both courses
make dl         # -> deep_learning/build/main.pdf
make ml         # -> machine_learning/build/main.pdf
make watch-dl   # rebuild on every save
make clean
```

## Adding notes

- **Fill an existing section:** edit `<course>/chapters/NN_chapter/MM_topic.tex`.
- **New section:** copy `common/section_template.tex` to `<course>/chapters/NN_chapter/MM_topic.tex`
  and add `\input{chapters/NN_chapter/MM_topic}` to `<course>/chapters/NN_chapter.tex`.
- **New chapter:** create `<course>/chapters/NN_chapter.tex` starting with `\chapter{Title}`
  (plus a `NN_chapter/` folder for its sections) and add `\include{chapters/NN_chapter}`
  to that course's `main.tex`.
- **Rename / reorder:** change the `\chapter{...}` / `\section{...}` titles and/or the order
  of `\include` / `\input` lines. File numbers are only for sorting.
- **Faster builds while writing:** uncomment `\includeonly{...}` in `main.tex` to compile
  only the chapters you list.
- **Citations:** add entries to `common/references.bib`, cite with `\citet{key}` / `\citep{key}`.

Available environments: `definition`, `theorem`, `lemma`, `proposition`, `corollary`,
`example`, `exercise`, `remark`, `note`, and boxes `keyidea`, `intuition`, `warning`.
Python code goes in `lstlisting`.

## Deep Learning: section 1.1

[Recap of Logistic Regression](deep_learning/chapters/01_logreg_to_mlp/01_recap_of_logistic_regression.tex)
contains first-principles derivations of log-odds, decision boundaries, Bernoulli
likelihood, binary cross-entropy, gradients, and convexity; worked examples,
cost-sensitive robot grasp decisions, and exercises with answers.

- [Compiled course notes](deep_learning/main.pdf)
- [NumPy implementation and figure generator](deep_learning/code/logreg_1_1.py)
- [Synthetic experiment results](deep_learning/figures/logreg_1_1/results.json)

The seven vector-PDF figures are committed, so `make dl` does not need Python.
To regenerate them, install NumPy and Matplotlib in your Python environment and run:

```bash
python deep_learning/code/logreg_1_1.py
make dl
```

The script uses seed 7, performs gradient/stability/curvature/descent checks, and
writes figures plus `results.json` under `deep_learning/figures/logreg_1_1/`.
The example is synthetic, not a measured robot benchmark. After rebuilding,
`deep_learning/build/main.pdf` is the fresh PDF; the root-level course PDF is a
committed snapshot.

## Deep Learning: section 1.2

[Cross Entropy, KL Divergence, and Maximum Likelihood](deep_learning/chapters/01_logreg_to_mlp/02_cross_entropy_kl_divergence_and_maximum_likelihood.tex)
derives surprise, entropy, coding bounds, cross-entropy, KL nonnegativity and
asymmetry, empirical maximum likelihood, and conditional classification losses.
It includes both lecture transfer questions, a Gaussian-mixture explanation of
KL direction, continuous-density caveats, a robot semantic-perception example,
and six exercises with worked answers.

- [NumPy experiments and seven figure generators](deep_learning/code/information_1_2.py)
- [Numerical results](deep_learning/figures/information_1_2/results.json)

```bash
python deep_learning/code/information_1_2.py
make dl
```

Requires NumPy and Matplotlib. The figures are committed; running Python is
optional for compiling the notes. Examples are constructed and the reverse-KL
Gaussian fit is a numerical search, with a quadrature convergence check.

## Deep Learning: sections 1.4 and 1.5

[Multilayer Perceptrons](deep_learning/chapters/01_logreg_to_mlp/04_multilayer_perceptrons.tex)
and [Matrix Dimensions](deep_learning/chapters/01_logreg_to_mlp/05_matrix_dimensions.tex)
derive hidden representations, the necessity of nonlinear activations, column-to-row
batch notation, broadcasting, and forward-pass time and memory costs. They include
all supplied lecture questions, a constructive XOR solution, a grasp-model resource
budget, and explicit distinctions between cached activations and inference buffers.

- [NumPy companion and four figure generators](deep_learning/code/mlp_1_4_1_5.py)
- [Checked numerical results](deep_learning/figures/mlp_1_4_1_5/results.json)

```bash
python deep_learning/code/mlp_1_4_1_5.py
make dl
```

The script checks scalar/vectorized and row/column equivalence, batch independence,
affine collapse, XOR predictions, and parameter/cache counts. Memory figures are
analytic array-element budgets, not measured runtime memory.

## Machine Learning: Chapter 3, first instalment

[Principles of Regularization](machine_learning/chapters/03_principles_of_regularization.tex)
now covers polynomial fitting, underfitting/overfitting, and the squared-loss
bias–variance decomposition. It includes a source map for the 14 supplied slide
screenshots, conditional-expectation clarifications, both cross-term cancellations,
worked examples, independent exercises, hints, solutions, and delayed review.
Regularization methods await the next lecture material.

- [Compiled ML reader](machine_learning/main.pdf)
- [NumPy experiment and figure generator](machine_learning/code/bias_variance.py)
- [Executed experiment results](machine_learning/figures/bias_variance/results.json)

```bash
python machine_learning/code/bias_variance.py
make ml
```

Figures are committed; Python is optional for building the reader. The experiment
uses fixed training inputs and repeated independent label noise, explicitly distinguished
from the slides' random-design experiment. Existing shared typography is preserved
(11 pt Latin Modern); it is not a Times New Roman export.
