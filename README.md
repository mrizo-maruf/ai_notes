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
