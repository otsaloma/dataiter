#!/bin/sh
rm -f *.df.csv
rm -f *.R.csv
echo "Generating data..."
python3 generate-df.py
Rscript generate.R
# Remove trailing zero decimals.
sed -ri "s/\.0*(,|$)/\1/g" *.csv
# Unify spelling of special values.
sed -ri "s/True/TRUE/g" *.csv
sed -ri "s/False/FALSE/g" *.csv
EXIT_STATUS=0
for NUM in $(ls *.df.csv | cut -d. -f1); do
    printf "%-23s" "Checking $NUM... "
    NLINES=$(diff -y --suppress-common-lines $NUM.df.csv $NUM.R.csv | wc -l)
    if [ $NLINES -gt 0 ]; then
        echo "$NLINES lines differ"
        EXIT_STATUS=1
    else
        echo "OK"
    fi
done
exit $EXIT_STATUS
