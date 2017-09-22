import sys

with open(sys.argv[1], 'r') as f:
    lines = f.read().rstrip('\n').split(' ')


with open('Q1.txt', 'w') as of:
    newList = []
    for word in lines:
        if word not in newList:
            newList.append(word)
            indx = newList.index(word)
            con = lines.count(word)
            of.write('{0} {1} {2}\n'.format(word, indx, con))
