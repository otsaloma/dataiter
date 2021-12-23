#!/bin/sh
rm -f *.ld.csv
rm -f *.R.csv
echo "Generating data..."
python3 generate-ld.py
Rscript generate.R
# Remove trailing zero decimals.
sed -ri "s/\.0*(,|$)/\1/g" *.csv
EXIT_STATUS=0
for NUM in $(ls *.ld.csv | cut -d. -f1); do
    printf "%-23s" "Checking $NUM... "
    NLINES=$(diff -y --suppress-common-lines $NUM.ld.csv $NUM.R.csv | wc -l)
    if [ $NLINES -gt 0 ]; then
        echo "$NLINES lines differ"
        EXIT_STATUS=1
    else
        echo "OK"
    fi
done
exit $EXIT_STATUS
