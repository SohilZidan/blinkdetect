#!/bin/bash

# TODO
# build third parties

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd third-party/ddfa

sh ./build.sh

cd SCRIPT_DIR