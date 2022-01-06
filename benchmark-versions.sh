#!/bin/sh

SCRIPT=benchmark-head.py
SCRIPT_ARGS="$@"
OUTPUT_FILE=benchmark-versions.csv
TMP_FILE=tmp.csv
VERSION=$(python3 -c "import dataiter; print(dataiter.__version__)")

benchmark() {
    VERSION=$1
    printf "\n$VERSION:\n"
    git checkout -q $VERSION
    ./$SCRIPT -o $TMP_FILE $SCRIPT_ARGS || true
    [ $VERSION = master ] && sed -i "s/$VERSION/HEAD/g" $TMP_FILE
    tail -n+2 $TMP_FILE >> $OUTPUT_FILE
    sed -i 's/"//g' $OUTPUT_FILE
}

set -e
rm -f $OUTPUT_FILE
echo "name,version,elapsed" > $OUTPUT_FILE
cp -fv benchmark.py $SCRIPT
for TAG in 0.12 0.13 0.14 0.15 0.16.1 0.17 0.18 0.19 0.20 0.21 0.22 0.23 0.24 0.25 0.26 0.27; do
    benchmark $TAG
done
benchmark master
rm -f $SCRIPT $TMP_FILE
