import sys, copy

def getValue(item):
	return float(item)

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
last = numbers.pop()
file = open('ans1.txt', 'w')
for element in numbers:
	file.write(element + ',')
file.write(last)
file.close()
