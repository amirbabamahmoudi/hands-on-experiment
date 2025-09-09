#!/bin/bash

# Battery Capacity Presentation Compilation Script
# This script compiles the LaTeX Beamer presentation to PDF

echo "🔄 Compiling Battery Capacity Presentation..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "❌ Error: pdflatex not found. Please install a LaTeX distribution (e.g., TeX Live, MiKTeX)"
    echo "   On macOS: brew install --cask mactex"
    echo "   On Ubuntu: sudo apt-get install texlive-full"
    exit 1
fi

# Compile the presentation (run twice for proper references)
echo "📝 First compilation pass..."
pdflatex -interaction=nonstopmode presentation.tex

echo "📝 Second compilation pass..."
pdflatex -interaction=nonstopmode presentation.tex

# Clean up auxiliary files
echo "🧹 Cleaning up auxiliary files..."
rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

# Check if PDF was created successfully
if [ -f "presentation.pdf" ]; then
    echo "✅ Success! Presentation compiled to: presentation.pdf"
    echo "📊 Opening presentation..."
    
    # Try to open the PDF with system default viewer
    if command -v open &> /dev/null; then
        open presentation.pdf  # macOS
    elif command -v xdg-open &> /dev/null; then
        xdg-open presentation.pdf  # Linux
    elif command -v start &> /dev/null; then
        start presentation.pdf  # Windows
    else
        echo "📁 PDF created successfully at: $(pwd)/presentation.pdf"
    fi
else
    echo "❌ Error: Failed to create presentation.pdf"
    echo "   Check the LaTeX log for errors"
    exit 1
fi

echo "🎉 Presentation ready for your battery capacity project!"
