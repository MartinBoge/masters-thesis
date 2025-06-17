.PHONY: main.pdf watch count clean

main.pdf:
	$(MAKE) count
	latexmk -pdf main.tex

watch:
	latexmk -pvc -pdf -view=none -shell-escape main.tex

count:
	./char_count.sh -s abstract
	./char_count.sh -s main

clean:
	rm -rf "$$(biber --cache)"
	git ls-files -i --exclude-standard -o | xargs rm -f