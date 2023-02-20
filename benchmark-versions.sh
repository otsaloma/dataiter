#!/bin/sh

SCRIPT=benchmark-head.py
SCRIPT_ARGS="$@"
OUTPUT_FILE=benchmark-versions.csv
TMP_FILE=tmp.csv

benchmark() {
    VERSION=$1
    printf "\n$VERSION:\n"
    git checkout -q $VERSION
    ./$SCRIPT -o $TMP_FILE --version=$VERSION $SCRIPT_ARGS || true
    tail -n+2 $TMP_FILE >> $OUTPUT_FILE
    sed -i 's/"//g' $OUTPUT_FILE
}

set -e
rm -f $OUTPUT_FILE
echo "name,version,elapsed" > $OUTPUT_FILE
cp -fv benchmark.py $SCRIPT
for VERSION in 0.40 master; do
    benchmark $VERSION
done
rm -f $SCRIPT $TMP_FILE
