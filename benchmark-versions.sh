#!/bin/sh

SCRIPT=benchmark-head.py
SCRIPT_ARGS="$@"
OUT_FILE=benchmark-versions.csv
TMP_FILE=tmp.csv

benchmark() {
    VERSION=$1
    printf "\n$VERSION:\n"
    git checkout -q $VERSION
    ./$SCRIPT -o $TMP_FILE --version=$VERSION $SCRIPT_ARGS || true
    tail -n+2 $TMP_FILE >> $OUT_FILE
    sed -i 's/"//g' $OUT_FILE
}

set -e
rm -f $OUT_FILE
echo "name,version,elapsed" > $OUT_FILE
cp -fv benchmark.py $SCRIPT
benchmark 1.0
benchmark master
rm -f $SCRIPT $TMP_FILE
