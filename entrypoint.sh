#!/bin/sh
set -eu

if [ "$#" -eq 0 ]; then
    exec python -m src.cli --query "¿Cuánto dura la garantía de una refrigeradora?"
fi

exec "$@"