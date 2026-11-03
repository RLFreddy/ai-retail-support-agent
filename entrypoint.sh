#!/bin/sh
set -eu

if [ "$#" -eq 0 ]; then
    exec tiendahogar ask "How long is the warranty on a refrigerator?"
fi

exec "$@"
