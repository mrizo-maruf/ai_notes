# latexmk config: PDF via pdflatex, build artifacts in build/
$pdf_mode = 1;
$out_dir  = 'build';
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 -file-line-error %O %S';
# \include writes sections/*.aux into out_dir, so that subfolder must exist
system("mkdir -p $out_dir/sections");
# Let bibtex find common/references.bib (latexmk runs from the course folder;
# an absolute path is needed because bibtex refuses '../' paths)
use Cwd 'abs_path';
$ENV{BIBINPUTS} = abs_path('../common') . ':' . ($ENV{BIBINPUTS} // '');
