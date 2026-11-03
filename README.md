# AI Course Notes

LaTeX notes for **Fundamentals of Deep Learning** and **Machine Learning**.
Each section lives in its own file, so notes can be added one at a time.

## Layout

```
common/
  preamble.tex          # packages, theorem envs, colored boxes, code style (shared)
  macros.tex            # math notation: \R, \E, \vec, \norm, \argmin, \loss, ...
  references.bib        # shared bibliography
  section_template.tex  # starting point for a new section
deep_learning/
  main.tex              # document root: title, TOC, list of \include{sections/...}
  sections/NN_topic.tex # one file per section
  figures/              # images for this course (\includegraphics{name.png})
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

- **Fill an existing section:** edit `<course>/sections/NN_topic.tex`.
- **New section:** copy `common/section_template.tex` to `<course>/sections/NN_topic.tex`
  and add `\include{sections/NN_topic}` to that course's `main.tex` in the right place.
- **Rename / reorder:** change the `\section{...}` title in the file and/or the order
  of `\include` lines in `main.tex`. File numbers are only for sorting.
- **Faster builds while writing:** uncomment `\includeonly{...}` in `main.tex`.
- **Citations:** add entries to `common/references.bib`, cite with `\citet{key}` / `\citep{key}`.

Available environments: `definition`, `theorem`, `lemma`, `proposition`, `corollary`,
`example`, `exercise`, `remark`, `note`, and boxes `keyidea`, `intuition`, `warning`.
Python code goes in `lstlisting`.
