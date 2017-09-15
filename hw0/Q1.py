import sys

fileName = sys.argv[1]

f = open(fileName, 'r')
lines = f.read().rstrip('\n').split(' ')
newLst = []
output = open('Q1.txt', 'w')
for i in lines:
    if i not in newLst:
        newLst.append(i)
        indx = newLst.index(i)
        con = lines.count(i)
        output.write('{0} {1} {2}\n'.format(i, indx, con))

f.close()
output.close()
