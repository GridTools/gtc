#!/bin/bash

set -e

REPO_ROOT=$(dirname $(dirname "$(readlink -e $0)"))
flake8 --select=D --ignore= --doctests --show-source ${REPO_ROOT}/src
