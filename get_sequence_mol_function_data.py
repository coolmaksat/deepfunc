#!/usr/bin/env python
# Filename: get_sequence_function_data.py
import sys

FILES = (
    # 'uniprot-go-0001071.txt',
    'uniprot-go-0003824.txt',
    'uniprot-go-0005198.txt',
    'uniprot-go-0005215.txt',
    'uniprot-go-0005488.txt',
    'uniprot-go-0060089.txt')
INVALID_ACIDS = set(['U', 'O', 'B', 'Z', 'J', 'X'])
LAMBDA = 24


def isOk(seq):
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


def main():

    min_len = sys.maxint
    data = list()
    for i in range(len(FILES)):
        file_name = FILES[i]
        data.append(dict())
        with open('data/molecular_functions/' + file_name, 'r') as f:
            next(f)
            for line in f:
                line = line.strip().split('\t')
                prot_id = line[0]
                exists = False
                for j in range(0, i):
                    if prot_id in data[j]:
                        del data[j][prot_id]
                        exists = True
                if exists:
                    continue
                seq = line[1]
                seq_len = len(seq)
                if seq_len > 24 and isOk(seq):
                    if min_len > seq_len:
                        min_len = seq_len
                    data[i][prot_id] = seq
    print 'Minimum length:', min_len

    with open('data/molecular_functions/uniprot-go-all-l.txt', 'w') as fall:
        for i in range(len(data)):
            n = 200000
            print len(data[i])
            for prot_id, seq in data[i].iteritems():
                fall.write(str(i) + '\t' + prot_id + '\t' + seq + '\n')
                n -= 1
                if n == 0:
                    break
if __name__ == '__main__':
    main()
