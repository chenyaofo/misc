#!/bin/sh

set -e

wget -q https://dl.chenyaofo.com/https://raw.githubusercontent.com/chenyaofo/fio-cdm/master/fio-cdm -O /tmp/fio-cdm
python /tmp/fio-cdm $1
rm -f /tmp/fio-cdm
