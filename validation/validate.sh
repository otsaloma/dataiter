#!/bin/sh
echo "Generating data..."
python3 generate.py
Rscript generate.R
# Remove trailing zero decimals.
sed -ri "s|\.0*(,\|$)|\1|g" *.csv
EXIT_STATUS=0
for NUM in $(ls *.py.csv | cut -d. -f1); do
    printf "Checking $NUM... "
    NLINES=$(diff -y --suppress-common-lines $NUM.py.csv $NUM.R.csv | wc -l)
    if [ $NLINES -gt 0 ]; then
        echo "$NLINES lines differ"
        EXIT_STATUS=1
    else
        echo "OK"
    fi
done
exit $EXIT_STATUS
