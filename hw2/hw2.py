# import sys
# import math
import math
import string


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    # Implementing vectors e,s as lists (arrays) of length 26
    # with p[0] being the probability of 'A' and so on
    e = [0] * 26
    s = [0] * 26

    with open('e.txt', encoding='utf-8') as f:
        for line in f:
            # strip: removes the newline character
            # split: split the string on space character
            char, prob = line.strip().split(" ")
            # ord('E') gives the ASCII (integer) value of character 'E'
            # we then subtract it from 'A' to give array index
            # This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char) - ord('A')] = float(prob)
    f.close()

    with open('s.txt', encoding='utf-8') as f:
        for line in f:
            char, prob = line.strip().split(" ")
            s[ord(char) - ord('A')] = float(prob)
    f.close()

    return (e, s)


def shred(filename):
    # Using a dictionary here. You may change this to any data structure of
    # your choice such as lists (X=[]) etc. for the assignment
    X = {}
    for char in string.ascii_uppercase:
        X[char] = 0

    with open(filename, encoding='utf-8') as f:
        fullText = f.read()
        fullText = fullText.upper()
        for chars in fullText:
            if chars.isalpha():
                X[chars] = X[chars] + 1
        dict(X)
        Q1 = "Q1\n"
        for element in X:
            Q1 = Q1 + element + " " + str(X[element]) + "\n"
    Q1 = Q1.strip("\n")
    print(Q1)
    return X


def findProbability():
    e, s = get_parameter_vectors()
    X = shred("letter.txt")
    xList = list(X.items())
    sumE = 0
    sumS = 0
    for i in range(26):
        ei = xList[i][1] * math.log(e[i])
        si = xList[i][1] * math.log(s[i])
        sumE += ei
        sumS += si
        if i == 0:
            print("Q2\n" + str('{0:.4f}'.format(ei, 4)) + "\n" + str('{0:.4f}'.format(si, 4)))

    fengl = math.log(0.6) + sumE
    fspan = math.log(0.4) + sumS

    print("Q3\n" + str('{0:.4f}'.format(fengl, 4)) + "\n" + str('{0:.4f}'.format(fspan, 4)))

    if fspan-fengl>=100:
        print("Q4\n0")
    elif fspan - fengl<=-100:
        print("Q4\n1")

    fEnglX = (1 / (1 + math.exp(fspan - fengl)))
    print("Q4\n" + str('{0:.4f}'.format(fEnglX, 4)))


# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!

findProbability()
