#!/bin/bash
FILES="data/molecular_functions/paac/*.txt"
ROOT="data/molecular_functions/paac/"

for f in $FILES; do
    bname=$(basename "$f")
    filename="${bname%.*}"
    #size=$(stat -c %s $ROOT$filename.txt)
    #limit=20000000
    #if [ -e "$ROOT""done/""$filename.txt" ]; then
    #    echo "Removing $filename"
    #    rm "$ROOT$filename.txt"
    #fi

    if [ -e $ROOT$filename.hdf5 ]; then
        echo "Moving $filename"
        mv "$ROOT$filename.txt" "$ROOT""done-80000"
        mv "$ROOT$filename.hdf5" "$ROOT""done-80000"
    fi
    #if [ $size -lt $limit ]; then
    #    echo "Moving $filename"
    #    mv "$ROOT$filename.txt" "$ROOT""outs"
    #fi
done
