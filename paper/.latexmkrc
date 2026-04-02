$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
$max_repeat = 5;
$force_mode = 1;
# Treat "Something's wrong--perhaps a missing \item" as non-fatal
# This is a known acmart bug with \resizebox inside figure*
$go_mode = 1;
