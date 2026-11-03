# Build course notes. PDFs land in <course>/build/main.pdf
COURSES := deep_learning machine_learning

.PHONY: all dl ml watch-dl watch-ml clean $(COURSES)

all: $(COURSES)

dl: deep_learning
ml: machine_learning

$(COURSES):
	cd $@ && latexmk -r ../.latexmkrc main.tex

# Recompile automatically on save
watch-dl:
	cd deep_learning && latexmk -r ../.latexmkrc -pvc main.tex
watch-ml:
	cd machine_learning && latexmk -r ../.latexmkrc -pvc main.tex

clean:
	for c in $(COURSES); do (cd $$c && latexmk -r ../.latexmkrc -C main.tex; rm -rf build); done
