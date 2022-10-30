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
for VERSION in 0.36; do
    benchmark $VERSION
done
benchmark master
rm -f $SCRIPT $TMP_FILE
