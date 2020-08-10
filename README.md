[![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-Ready--to--Code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/eth-cscs/eve_toolchain)

Eve: A framework for implementing DSL toolchains in Python
==========================================================

Installation instructions
-------------------------

Eve contains a standard `setup.py` installation script which can be
used to install the package with *pip*. The simplest way to start working
with Eve is using the provided scripts in the `scripts/` folder. The standard
`activate` script will properly initialize the virtual environment if it does
not exist yet:

    git clone https://github.com/egparedes/eve.git
    cd eve
    source scripts/activate


If you know how to deal with virtual environments yourself, create first as usual
a new environment for your project, and then clone the repository and install Eve
(use _editable_ mode if you plan to contribute to Eve):

    # Create the virtual environment
    python -m venv .eve.venv
    source .eve.venv/bin/activate
    pip install --upgrade pip setuptools wheel

    # Clone the repository
    git clone https://github.com/egparedes/eve.git

    # Install the Python package directly from the local repository
    # adding the '-e' flag to get an editable installation
    pip install -e ./eve

    # Finally, install the additional development tools
    pip install -r ./eve/requirements_dev.txt


Development instructions
-------------------------

`Tox`, `py.test` and `pre-commit` (running several checks, including `black` formatter and `flake8`) are already configured and should run out of the box after installing the required development tools in `requirements_dev.txt`:

    # Generate HTML documentation:
    cd docs && make html

    # Execute all the tests in the `tests` folder with py.test:
    py.test -v ./tests

    # Install pre-commit git hooks:
    pre-commit install

    # Or, alternatively, execute all pre-commit checks manually:
    pre-commit run --all

Code editors supporting the [Editorconfig](http://editorconfig.org) standard should be automatically configured (settings in `.editorconfig`).
