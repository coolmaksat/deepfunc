#!/usr/bin/env python
"""
Instead of using the conventional 20-D amino acid composition to represent the
sample of a protein, Prof. Kuo-Chen Chou proposed the pseudo amino acid (PseAA)
composition in order for inluding the sequence-order information. Based on the
concept of Chou's pseudo amino acid composition, the server PseAA was designed
in a flexible way, allowing users to generate various kinds of pseudo amino
acid composition for a given protein sequence by selecting different parameters
and their combinations. This module aims at computing two types of PseAA
descriptors: Type I and Type II. You can freely use and distribute it. If you
have any problem, you could contact with us timely.

References:
[1]: Kuo-Chen Chou. Prediction of Protein Cellular Attributes Using
Pseudo-Amino Acid Composition. PROTEINS: Structure, Function, and Genetics,
2001, 43: 246-255.
[2]: http://www.csbio.sjtu.edu.cn/bioinf/PseAAC/
[3]: http://www.csbio.sjtu.edu.cn/bioinf/PseAAC/type2.htm
[4]: Kuo-Chen Chou. Using amphiphilic pseudo amino acid composition to predict
enzyme subfamily classes. Bioinformatics, 2005,21,10-19.
Authors: Dongsheng Cao and Yizeng Liang.
Date: 2012.9.2
Email: oriental-cds@163.com
The hydrophobicity values are from JACS, 1962, 84: 4240-4246. (C. Tanford).
The hydrophilicity values are from PNAS, 1981, 78:3824-3828
(T.P.Hopp & K.R.Woods). The side-chain mass for each of the 20 amino acids. CRC
Handbook of Chemistry and Physics, 66th ed., CRC Press, Boca Raton,
Florida (1985). R.M.C. Dawson, D.C. Elliott, W.H. Elliott, K.M. Jones,
Data for Biochemical Research 3rd ed.,
Clarendon Press Oxford (1986).

Note:
The code was refactored to improve readability and performance by
Maxat Kulmanov, maxat.kulmanov@kaust.edu.sa
"""

import math

AALETTER = [
    "A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

HYDROPHOBICITY = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85,
    "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50,
    "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81,
    "Y": 0.26, "V": 1.08}

HYDROPHILICITY = {
    "A": -0.5, "R": 3.0, "N": 0.2, "D": 3.0, "C": -1.0, "Q": 0.2, "E": 3.0,
    "G": 0.0, "H": -0.5, "I": -1.8, "L": -1.8, "K": 3.0, "M": -1.3, "F": -2.5,
    "P": 0.0, "S": 0.3, "T": -0.4, "W": -3.4, "Y": -2.3, "V": -1.5}

RESIDUEMASS = {
    "A": 15.0, "R": 101.0, "N": 58.0, "D": 59.0, "C": 47.0, "Q": 72.0,
    "E": 73.0, "G": 1.000, "H": 82.0, "I": 57.0, "L": 57.0, "K": 73.0,
    "M": 75.0, "F": 91.0, "P": 42.0, "S": 31.0, "T": 45.0, "W": 130.0,
    "Y": 107.0, "V": 43.0}

PK1 = {
    "A": 2.35, "C": 1.71, "D": 1.88, "E": 2.19, "F": 2.58, "G": 2.34,
    "H": 1.78, "I": 2.32, "K": 2.20, "L": 2.36, "M": 2.28, "N": 2.18,
    "P": 1.99, "Q": 2.17, "R": 2.18, "S": 2.21, "T": 2.15, "V": 2.29,
    "W": 2.38, "Y": 2.20}

PK2 = {
    "A": 9.87, "C": 10.78, "D": 9.60, "E": 9.67, "F": 9.24, "G": 9.60,
    "H": 8.97, "I": 9.76, "K": 8.90, "L": 9.60, "M": 9.21, "N": 9.09,
    "P": 10.6, "Q": 9.13, "R": 9.09, "S": 9.15, "T": 9.12, "V": 9.74,
    "W": 9.39, "Y": 9.11}

PI = {
    "A": 6.11, "C": 5.02, "D": 2.98, "E": 3.08, "F": 5.91, "G": 6.06,
    "H": 7.64, "I": 6.04, "K": 9.47, "L": 6.04, "M": 5.74, "N": 10.76,
    "P": 6.30, "Q": 5.65, "R": 10.76, "S": 5.68, "T": 5.60, "V": 6.02,
    "W": 5.88, "Y": 5.63}

# HYDROPHOBICITY = [
#     0.62, 0.0, 0.29, -0.9, -0.74, 1.19, 0.48, -0.4, 1.38, 0.0, -1.5, 1.06,
#     0.64, -0.78, 0.0, 0.12, -0.85, -2.53, -0.18, -0.05, 0.0, 1.08, 0.81, 0.0,
#     0.26, 0.0]
# HYDROPHILICITY = [
#     -0.5, 0.0, -1.0, 3.0, 3.0, -2.5, 0.0, -0.5, -1.8, 0.0, 3.0, -1.8, -1.3,
#     0.2, 0.0, 0.0, 0.2, 3.0, 0.3, -0.4, 0.0, -1.5, -3.4, 0.0, -2.3, 0.0]
# RESIDUEMASS = [
#     15.0, 0.0, 47.0, 59.0, 73.0, 91.0, 1.0, 82.0, 57.0, 0.0, 73.0, 57.0,
#     75.0, 58.0, 0.0, 42.0, 72.0, 101.0, 31.0, 45.0, 0.0, 43.0, 130.0, 0.0,
#     107.0, 0.0]
# PK1 = [
#     2.35, 0.0, 1.71, 1.88, 2.19, 2.58, 2.34, 1.78, 2.32, 0.0, 2.2, 2.36,
#     2.28, 2.18, 0.0, 1.99, 2.17, 2.18, 2.21, 2.15, 0.0, 2.29, 2.38, 0.0,
#     2.2, 0.0]
# PK2 = [
#     9.87, 0.0, 10.78, 9.6, 9.67, 9.24, 9.6, 8.97, 9.76, 0.0, 8.9, 9.6,
#     9.21, 9.09, 0.0, 10.6, 9.13, 9.09, 9.15, 9.12, 0.0, 9.74, 9.39, 0.0,
#     9.11, 0.0]
# PI = [
#     6.11, 0.0, 5.02, 2.98, 3.08, 5.91, 6.06, 7.64, 6.04, 0.0, 9.47, 6.04,
#     5.74, 10.76, 0.0, 6.3, 5.65, 10.76, 5.68, 5.6, 0.0, 6.02, 5.88, 0.0,
#     5.63, 0.0]


def mean(a):
    """
    The mean value of the list data.
    Usage:
        result = mean(array)
    """
    return sum(a)/len(a)


def std(a, ddof=0):
    """
    The standard deviation of the list data.
    Usage:
        result = std(array)
    """
    m = mean(a)
    temp = [math.pow(i-m, 2) for i in a]
    res = math.sqrt(sum(temp) / (len(a) - ddof))
    return res


def normalize_aap(aap):
    """
    All of the amino acid indices are centralized and
    standardized before the calculation.
    Usage:
        result = normalize_aap(aap)
    Input: aap is a dict form containing the properties of 20 amino acids.
    Output: result is the a dict form containing the normalized properties
    of 20 amino acids.
    """

    if len(aap) != 20:
        raise Exception('Invalid number of Amino acids!')

    res = dict()
    aap_mean = mean(aap.values())
    aap_std = std(aap.values())
    for key, value in aap.iteritems():
        res[key] = (value - aap_mean) / aap_std
    return res


# Type I descriptors
# Pseudo-Amino Acid Composition descriptors


def get_correlation_function(norm_aap, ri, rj):
    """
    Computing the correlation between two given amino acids using the above
    three properties.
    Usage:
        result = get_correlation_function(aap, ri, rj)
    Input:
        aap are properties
        ri and rj are the amino acids, respectively.
    Output:
        result is the correlation value between two amino acids.
    """
    if len(norm_aap) == 0:
        raise Exception('No amino acid properties provided!')
    theta = 0.0
    for i in range(len(norm_aap)):
        temp = norm_aap[i][ri] - norm_aap[i][rj]
        theta += temp * temp
    theta = theta / len(norm_aap)
    return theta


def get_sequence_order_correlation_factor(norm_aap, protein_sequence, k=1):
    """
    Computing the sequence order correlation factor with gap equal to k
    based on [HYDROPHOBICITY, HYDROPHILICITY, RESIDUEMASS].
    Usage:
        result =get_sequence_order_correlation_factor(protein_sequence, k)
    Input:
        protein is a pure protein sequence.
        k is the gap.
    Output:
        result is the correlation factor value with the gap equal to k.
    """
    sequence_length = len(protein_sequence)
    res = []
    for i in range(sequence_length - k):
        aa1 = protein_sequence[i]
        aa2 = protein_sequence[i + k]
        res.append(get_correlation_function(norm_aap, aa1, aa2))
    result = round(sum(res) / (sequence_length - k), 3)
    return result


def get_aa_composition(protein_sequence):

    """
    Calculate the composition of Amino acids for a given protein sequence.
    Usage:
        result = calculate_aa_composition(protein)
    Input:
        protein is a pure protein sequence.
    Output:
        result is a dict form containing the composition of 20 amino acids.
    """
    sequence_length = len(protein_sequence)
    result = {}
    for i in AALETTER:
        result[i] = float(protein_sequence.count(i)) / sequence_length * 100
    return result


def get_pseudo_aac1(protein_sequence, lamda=10, weight=0.05, norm_aap=[]):
    """
    Computing the first 20 of type I pseudo-amino acid compostion descriptors
    based on [HYDROPHOBICITY, HYDROPHILICITY, RESIDUEMASS].
    """
    rightpart = 0.0
    for i in range(lamda):
        rightpart = rightpart + get_sequence_order_correlation_factor(
            norm_aap, protein_sequence, k=i+1)
    aac = get_aa_composition(protein_sequence)

    result = list()
    temp = 1 + weight * rightpart
    for index, i in enumerate(AALETTER):
        result.append(round(aac[i] / temp, 3))
    return result


def get_pseudo_aac2(
        protein_sequence, lamda=10, weight=0.05, norm_aap=[]):
    """
    Computing the last lamda of type I pseudo-amino acid compostion descriptors
    based on [HYDROPHOBICITY, HYDROPHILICITY, RESIDUEMASS].
    """

    rightpart = []
    for i in range(lamda):
        rightpart.append(get_sequence_order_correlation_factor(
            norm_aap, protein_sequence, k=i+1))

    result = list()
    temp = 1 + weight * sum(rightpart)
    for index in range(20, 20 + lamda):
        result.append(round(
            weight * rightpart[index - 20] / temp * 100, 3))
    return result


def get_pseudo_aac(
        protein_sequence, lamda=10,
        weight=0.05, aap=[HYDROPHOBICITY, HYDROPHILICITY, RESIDUEMASS]):
    """
    Computing all of type I pseudo-amino acid compostion descriptors based on
    three given properties. Note that the number of PAAC strongly depends on
    the lamda value. if lamda = 20, we can obtain 20 + 20 = 40 PAAC
    descriptors. The size of these values depends on the choice of lamda
    and weight simultaneously.
    AAP = [HYDROPHOBICITY, HYDROPHILICITY, RESIDUEMASS]
    Usage:
        result = get_pseudo_aac(protein, lamda, weight)
    Input:
        protein is a pure protein sequence.
        lamda factor reflects the rank of correlation and is a non-negative
        integer, such as 15.
        Note that:
        (1) lamda should NOT be larger than the length of input protein
            sequence;
        (2) lamda must be non-Negative integer, such as 0, 1, 2, ...;
        (3) when lamda = 0, the output of PseAA server is the 20-D amino acid
        composition.
        weight factor is designed for the users to put weight on the additional
        PseAA components with respect to the conventional AA components.
        The user can select any value within the region from 0.05 to 0.7 for
        the weight factor.
    Output:
        result is a dict form containing calculated 20 + lamda PAAC
        descriptors.
    """
    norm_aap = list()
    for item in aap:
        norm_aap.append(normalize_aap(item))
    res1 = get_pseudo_aac1(
        protein_sequence, lamda=lamda, weight=weight, norm_aap=norm_aap)
    res2 = get_pseudo_aac2(
        protein_sequence, lamda=lamda, weight=weight, norm_aap=norm_aap)
    res = res1 + res2
    return res


# Type II descriptors
# Amphiphilic Pseudo-Amino Acid Composition descriptors

NORM_HYDROPHOBICITY = normalize_aap(HYDROPHOBICITY)
NORM_HYDROPHILICITY = normalize_aap(HYDROPHILICITY)
NORM_RESIDUEMASS = normalize_aap(HYDROPHILICITY)
NORM_PK1 = normalize_aap(PK1)
NORM_PK2 = normalize_aap(PK2)
NORM_PI = normalize_aap(PI)

APAAC_CORRELATION = dict()


def get_correlation_function_apaac(ri, rj):
    """
    Computing the correlation between two given amino acids using the above six
    properties for APAAC (type II PseAAC).
    Usage:
        result = get_correlation_function_for_apaac(ri, rj)
    Input:
        ri and rj are the amino acids, respectively.
    Output:
        result is the correlation value between two amino acids.
    """
    if ri in APAAC_CORRELATION and rj in APAAC_CORRELATION[ri]:
        return APAAC_CORRELATION[ri][rj]
    if rj in APAAC_CORRELATION and ri in APAAC_CORRELATION[rj]:
        return APAAC_CORRELATION[rj][ri]
    theta1 = NORM_HYDROPHOBICITY[ri] * NORM_HYDROPHOBICITY[rj]
    theta2 = NORM_HYDROPHILICITY[ri] * NORM_HYDROPHILICITY[rj]
    theta3 = NORM_RESIDUEMASS[ri] * NORM_RESIDUEMASS[rj]
    theta4 = NORM_PK1[ri] * NORM_PK1[rj]
    theta5 = NORM_PK2[ri] * NORM_PK2[rj]
    theta6 = NORM_PI[ri] * NORM_PI[rj]
    if ri not in APAAC_CORRELATION:
        APAAC_CORRELATION[ri] = dict()
        APAAC_CORRELATION[ri][rj] = dict()
    if rj not in APAAC_CORRELATION:
        APAAC_CORRELATION[rj] = dict()
        APAAC_CORRELATION[rj][ri] = dict()
    APAAC_CORRELATION[ri][rj] = (theta1, theta2, theta3, theta4, theta5, theta6)
    APAAC_CORRELATION[rj][ri] = (theta1, theta2, theta3, theta4, theta5, theta6)
    return APAAC_CORRELATION[ri][rj]


def get_sequence_order_correlation_factor_apaac(protein_sequence, k=1):
    """
    Computing the Sequence order correlation factor with gap equal to k
    based on [HYDROPHOBICITY, HYDROPHILICITY] for APAAC (type II PseAAC).
    Usage:
        result = get_sequence_order_correlation_factor_apaac(protein, k)
    Input:
        protein is a pure protein sequence.
        k is the gap.
    Output:
        result is the correlation factor value with the gap equal to k.
    """

    sequence_length = len(protein_sequence)
    hydrophobicity = []
    hydrophilicity = []
    residuemass = []
    pk1 = []
    pk2 = []
    pi = []
    for i in range(sequence_length - k):
        aa1 = protein_sequence[i]
        aa2 = protein_sequence[i + k]
        temp = get_correlation_function_apaac(aa1, aa2)
        hydrophobicity.append(temp[0])
        hydrophilicity.append(temp[1])
        residuemass.append(temp[2])
        pk1.append(temp[3])
        pk2.append(temp[4])
        pi.append(temp[5])
    result = []
    result.append(sum(hydrophobicity) / (sequence_length - k))
    result.append(sum(hydrophilicity) / (sequence_length - k))
    result.append(sum(residuemass) / (sequence_length - k))
    result.append(sum(pk1) / (sequence_length - k))
    result.append(sum(pk2) / (sequence_length - k))
    result.append(sum(pi) / (sequence_length - k))
    return result


def get_apseudo_aac(protein_sequence, lamda=24, weight=0.5):
    """
    Computing all of type II pseudo-amino acid compostion descriptors based on
    three given properties. Note that the number of PAAC strongly depends on
    the lamda value. if lamda = 20, we can obtain 20 + 20 = 40 PAAC
    descriptors. The size of these values depends on the choice of lamda
    and weight simultaneously.
    Usage:
        result = get_apseudo_aac(protein, lamda, weight)
    Input:
        protein is a pure protein sequence.
        lamda factor reflects the rank of correlation and is a non-negative
        integer, such as 15.
        Note that:
        (1) lamda should NOT be larger than the length of input protein
            sequence;
        (2) lamda must be non-Negative integer, such as 0, 1, 2, ...;
        (3) when lamda = 0, the output of PseAA server is the 20-D amino acid
        composition.
        weight factor is designed for the users to put weight on the additional
        PseAA components with respect to the conventional AA components.
        The user can select any value within the region from 0.05 to 0.7 for
        the weight factor.
    Output:
        result is a dict form containing calculated 20 + lamda PAAC
        descriptors.

    """
    total = 0.0
    order_cor_factors = list()
    for i in range(lamda):
        order_cor_factors.append(get_sequence_order_correlation_factor_apaac(
                protein_sequence, k=i+1))
        total += sum(order_cor_factors[i])
    aac = get_aa_composition(protein_sequence)
    result = list()
    temp = 1 + weight * total
    if temp == 0.0:
        return []
    for index, i in enumerate(AALETTER):
        result.append(round(aac[i] / temp, 3))

    rightpart = []
    for i in range(lamda):
        temp = order_cor_factors[i]
        rightpart.append(temp[0])
        rightpart.append(temp[1])
        rightpart.append(temp[2])
        rightpart.append(temp[3])
        rightpart.append(temp[4])
        rightpart.append(temp[5])

    temp = 1 + weight * sum(rightpart)
    if temp == 0.0:
        return []
    for index in range(6 * lamda):
        result.append(round(
            weight * rightpart[index] / temp * 100, 3))

    return result


def load_data():
    data = list()
    with open('data/uniprot-swiss-mol-func.txt') as f:
        for line in f:
            line = line.strip().split('\t')
            prot_id = line[0]
            seq = line[1]
            data.append((prot_id, seq))
    return data


def main(*args, **kwargs):

    data = load_data()
    print 'Data has been loaded!'
    with open(
            'data/uniprot-swiss-mol-func-paac.txt',
            'w', 1073741824) as f:
        for prot_id, seq in data:
            f.write(prot_id)
            paac = get_apseudo_aac(seq, lamda=24)
            for p in paac:
                f.write(' ' + str(p))
            f.write('\n')

if __name__ == '__main__':
    main()
