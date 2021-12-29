#!/bin/bash
for ARG; do
    ARG="${ARG//_/-}"
    touch blocks/$ARG-dplyr.R
    touch blocks/$ARG-pandas.py
    touch blocks/$ARG-dataiter.py
done
