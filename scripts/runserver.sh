#!/bin/bash
set -euo pipefail
IFS=$'\n\t'


export FLASK_APP=app.py
export FLASK_ENV=production

for arg in "$@"
do
    case $arg in
        -d|--debug)
        FLASK_ENV=development
        shift
        ;;
    esac
done

export VOCAB_ENT=$1
export VOCAB_REL=$2
export MODEL_DIR=$3
export PATH_FILE=${4:-""}

pipenv run flask run --host 0.0.0.0