#!/bin/bash
for lvl in {1..10}; do
    DIRS="data/fofe/level_$lvl/GO:00*"

    for d in $DIRS; do
        FILES="$d/GO*.hdf5"
        parent=$(basename $d)
        for f in $FILES; do
            if [ -e $f ]; then
                bname=$(basename "$f")
                filename="${bname%.*}"
                size=$(stat -c%s "$f")
                if [ "$size" = "800" ]; then
                    echo "The $f is $size bytes. Deleting..."
                    rm $f
                fi
                echo $filename
            fi
        done
    done
done
