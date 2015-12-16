#!/usr/bin/env python
# Filename: get_sequence_function_data.py
from pseudo_aac import get_apseudo_aac

FILES = (
    'uniprot-go-0001071',
    'uniprot-go-0003824',
    'uniprot-go-0005198',
    'uniprot-go-0005215',
    'uniprot-go-0005488',
    'uniprot-go-0009055',
    'uniprot-go-0060089',
    'uniprot-go-0098772')
INVALID_ACIDS = set(['U', 'O', 'B', 'Z', 'J', 'X'])
LAMBDA = 24


def isOk(seq):
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


def main():

    for i in range(len(FILES)):
        file_name = FILES[i]
        with open('data/molecular_functions/' + file_name + '.txt', 'r') as f:
            next(f)
            with open(
                'data/molecular_functions/' + file_name + '.paac.txt', 'w'
            ) as fall:
                for line in f:
                    line = line.strip().split('\t')
                    prot_id = line[0]
                    seq = line[1]
                    seq_len = len(seq)
                    if seq_len > 24 and isOk(seq):
                        fall.write(prot_id)
                        paac = get_apseudo_aac(seq, lamda=LAMBDA)
                        for x in paac:
                            fall.write(' ' + str(x))
                        fall.write('\n')

if __name__ == '__main__':
    main()
