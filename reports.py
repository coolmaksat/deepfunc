#!/usr/bin/env python

ROOT = 'data/molecular_functions/paac/'
FILES = ('reports.txt', 'reports-rob.txt',)


def load_data():
    data = dict()
    for filename in FILES:
        with open(ROOT + filename, 'r') as f:
            for line in f:
                go_id = line.strip().split()[3]
                next(f)
                next(f)
                next(f)
                next(f)
                next(f)
                line = next(f)
                f_score = line.strip().split()[5]
                data[go_id] = f_score
                next(f)
    return data


def main():
    data = load_data()
    with open(ROOT + 'all-reports.txt', 'w') as f:
        for go_id, f_score in data.iteritems():
            f.write(go_id + '\t' + f_score + '\n')

if __name__ == '__main__':
    main()
