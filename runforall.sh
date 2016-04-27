#!/bin/bash
for lvl in {1..10}; do
    DIRS="data/fofe/level_$lvl/GO:00*"

    for d in $DIRS; do
        FILES="$d/GO*.pkl"
        parent=$(basename $d)
        for f in $FILES; do
            bname=$(basename "$f")
            filename="${bname%.*}"
            if [ ! -e "$d/$filename.hdf5" ]; then
                echo "Running neural network for level $lvl $parent $filename"
                THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python nn_fofe.py $parent $filename $lvl
                THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python gen_next_level_data.py $parent $filename $lvl
                python gen_next_level_training.py $parent $filename $lvl
                sleep 1s
            fi
        done
    done
done
