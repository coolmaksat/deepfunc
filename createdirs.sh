#!/bin/bash
FILES="data/swiss/level_2/data/GO:00*.txt"
ROOT="data/swiss/level_2/"
for f in $FILES; do
    bname=$(basename "$f")
    filename="${bname%.*}"
    echo "Creating dir $filename"
    mkdir $ROOT$filename
done
