#!/bin/sh

# Default values
section=""
csl="apa"
verbose=false

# Function to show usage/help message
usage() {
  echo "Usage: $0 --section <section> --csl <csl>"
  echo
  echo "Options:"
  echo "  --help          Show this help message"
  echo "  --section, -s   Specify the section to process, marked with '%COUNT:<section>' and '%COUNT:end<section>' in your tex file. (Required)"
  echo "                  Example: '%COUNT:Introduction' and '%COUNT:endIntroduction' for the 'Introduction' section."
  echo "  --csl, -c       Specify the CSL file (default: apa)"
  echo "  --verbose, -v   Show the final count  "
  echo
  exit 1
}

# If no arguments are provided, show usage
if [ $# -eq 0 ]; then
  usage
fi

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help)
      usage ;;
    --section|-s)
      section="$2"
      shift 2 ;;
    --csl|-c)
      csl="$2"
      shift 2 ;;
    --verbose|-v)
      verbose=true
      shift ;;
    *)
      echo "Unknown option: $1"
      usage ;;
  esac
done

# Check if section is null or empty
if [ -z "$section" ]; then
  echo "Error: Section argument (--section or -s) is required."
  exit 1
fi

# Check if CSL file exists
if [ ! -f "$csl.csl" ]; then
  echo "Error: CSL file '$csl.csl' not found."
  echo "Tried to find it in the current directory: $(pwd)"
  echo "You can download citation styles at https://github.com/citation-style-language/styles."
  exit 1
fi

# Check if main.tex exists
if [ ! -f "main.tex" ]; then
  echo "Error: main.tex not found in current directory."
  exit 1
fi

# Check if sections directory exists
if [ ! -d "sections" ]; then
  echo "Error: sections directory not found in current directory."
  exit 1
fi

# Create temporary dir and ensure cleanup no matter what
temp_dir=$(mktemp -d)
trap "rm -rf $temp_dir" EXIT

# Convert \include to \input in main.tex and all section files
sed 's/\\include{/\\input{/g' main.tex > "$temp_dir/main_preprocessed.tex"

# Process all .tex files in sections directory if it exists
mkdir -p "$temp_dir/sections"
for tex_file in sections/*.tex; do
  if [ -f "$tex_file" ]; then
    filename=$(basename "$tex_file")
    sed 's/\\include{/\\input{/g' "$tex_file" > "$temp_dir/sections/$filename"
  fi
done

# Expand tex (add tex content coming via \input and \include)
latexpand --keep-comments "$temp_dir/main_preprocessed.tex" > "$temp_dir/expanded.tex" 2>/dev/null

# Filter document (only keep content inside document that is between %COUNT:section and %COUNT:endsection)
sed -r -n "1,/\\begin{document}/p; /%COUNT:${section}/,/%COUNT:end${section}/p; /\\end{document}/p" "$temp_dir/expanded.tex" > "$temp_dir/filtered_section.tex"

# Count tables and figures (ignoring line comments)
figure_count=$(grep -c -E '^[[:space:]]*[^%]*\\begin{figure\*?}' "$temp_dir/filtered_section.tex")
table_count=$(grep -c -E '^[[:space:]]*[^%]*\\begin{table\*?}'  "$temp_dir/filtered_section.tex")

# Filter document (remove table and figure elements)
sed -E '/\\begin{table\*?}/,/\\end{table\*?}/d; /\\begin{figure\*?}/,/\\end{figure\*?}/d' "$temp_dir/filtered_section.tex" > "$temp_dir/filtered_content.tex"

# Insert unique marker before \end{document} because pandoc is unfortunately appending a bibliography that we have to strip
marker="____PANDOC_MARKER____"
sed "/\\\end{document}/i\\
\\\newline $marker
" "$temp_dir/filtered_content.tex" > "$temp_dir/filtered_content_with_marker.tex"

# Compile to plain text
pandoc "$temp_dir/filtered_content_with_marker.tex" --citeproc --csl "$csl.csl" -t plain -o "$temp_dir/content_with_refs.txt" --quiet

# Strip everything from and including the pandoc marker
sed "/$marker/,\$d" "$temp_dir/content_with_refs.txt" > "$temp_dir/content_with_refs_and_marker_stripped.txt"

# Remove linebreaks
tr -d '\n' < "$temp_dir/content_with_refs_and_marker_stripped.txt" > "$temp_dir/content_for_count.txt"

# Count characters
char_count=$(wc -m < "$temp_dir/content_for_count.txt")
final_count=$(($char_count + ($table_count + $figure_count) * 800))

# Output the results
if [ "$verbose" = true ]; then
  echo "Character count for '${section}' section: $final_count"
fi

printf "%'d" "$final_count" > ".char_count.${section}.txt"
