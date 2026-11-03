#!/bin/sh
set -eu

if [ "$#" -eq 0 ]; then
    exec tiendahogar ask "¿Cuánto dura la garantía de una refrigeradora?"
fi

exec "$@"