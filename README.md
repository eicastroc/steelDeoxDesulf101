# steelDeoxDesulf101

Principles of steel deoxidation and desulfurization with pedagogic intents.

## Contents

### English

Coming soon...

### French

Coming soon...

### Spanish

1. [Fundamentos de desoxidación de aceros](src/es/Deox.py)
2. [Fundamentos de desulfuración de aceros](src/es/Desulf.py)

## Usage

To run the notebooks you need an install of the [Python](https://www.python.org/) programming language.

To setup the environment in a reproducible way, first create a local environment by running in command line:

```bash
# Install the `virtualenv` package
pip install virtualenv

# Create a local virtual environment
python -m virtualenv venv

# Activate the environment
./venv/Scripts/activate

# Install the project requirements
pip install -r requirements.txt 
```

Then simply launch `jupyter-notebook` from a terminal and you should be ready to work in your browser.

For those willing to work within VSCode, you can install extensions `Python` and `Jupyter` (both from Microsoft).

The Python files above are written with the percent `%%` format allowing for execution as notebooks.

## Creating book

Implemented with [Jupyter Book](https://jupyterbook.org/).

To build the project run the following from this directory

```
jupyter-book build --all --path-output=docs src
```

To generate a PDF build run

```
jupyter-book build --all --path-output=docs --builder pdflatex src
```

Relevant documentation:
- [Configuration file](https://jupyterbook.org/customize/config.html)
- [Table of contents](https://jupyterbook.org/customize/toc.html)

## Citing

For citing this work, please consider [this](CITATION.bib) bibliographic entry.
