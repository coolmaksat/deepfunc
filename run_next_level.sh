#!/bin/bash
FILES="data/swiss/level_2/data/GO:00*.txt"
for f in $FILES; do
    bname=$(basename "$f")
    filename="${bname%.*}"
    echo "Running for $filename"
    ./gen_next_level_training.py $filename
done
