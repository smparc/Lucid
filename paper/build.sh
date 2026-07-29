#!/usr/bin/env bash
# Build the paper.
#
#   ./build.sh            compile paper.tex -> paper.pdf
#   ./build.sh --clean    remove LaTeX intermediates
#
# results_ablation.tex is generated from the experiment output by
# make_results_table.py; it is committed so the paper builds without rerunning
# the experiments.

set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--clean" ]]; then
    rm -f ./*.aux ./*.log ./*.out ./*.toc ./*.pdf
    echo "cleaned"
    exit 0
fi

# Two passes so the table of contents and cross-references resolve.
for pass in 1 2; do
    pdflatex -interaction=nonstopmode -halt-on-error paper.tex > "build${pass}.log" 2>&1 || {
        echo "LaTeX failed on pass ${pass}; last errors:"
        grep -E "^!|l\.[0-9]+" "build${pass}.log" | head -20
        exit 1
    }
done

rm -f build1.log build2.log
echo "Built paper.pdf ($(du -h paper.pdf | cut -f1))"
