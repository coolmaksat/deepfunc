#!/bin/bash
FILES="data/swiss/level_1/GO:00*.txt"
ROOT="data/swiss/level_1/"

for f in $FILES; do
    bname=$(basename "$f")
    filename="${bname%.*}"
    if [ ! -e "$ROOT$filename.hdf5" ]; then
        echo "Running neural network for $filename"
        THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python nn_sequence_paac.py $filename
        sleep 5s
    fi
done
