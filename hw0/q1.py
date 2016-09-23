import sys, copy
import numpy as np

def getValue(item):
	return float(item)

# print(sys.argv)
index = int(sys.argv[1])
filename = sys.argv[2]
with open(filename) as inf:
    data = []
    for line in inf:
        line = line.split()
        data.append(line)

numbers = []
for element in data:
	numbers.append(element[index])
numbers = sorted(numbers, key=getValue)
print(numbers)
last = numbers.pop()
file = open('ans1.txt', 'w')
for element in numbers:
	file.write(element + ',')
file.write(last)
file.close()
