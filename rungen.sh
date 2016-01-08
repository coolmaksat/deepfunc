#!/bin/bash
FILES="data/swiss/level_1/GO:00*.txt"

for f in $FILES; do
    bname=$(basename "$f")
    filename="${bname%.*}"
    echo "Running generation for $filename"
    THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python gen_next_level_data.py $filename
    sleep 5s
done
