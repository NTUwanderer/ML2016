CFLAGS = -Wall -O3

run: main.o table.o problem.o util.h
	g++ -o $@ main.o table.o problem.o $(CFLAGS)

main.o: main.cpp
	g++ -c main.cpp $(CFLAGS)

table.o: table.cpp
	g++ -c table.cpp $(CFLAGS)

problem.o: problem.cpp
	g++ -c problem.cpp $(CFLAGS)

# util.o: util.cpp
# 	g++ -c util.cpp $(CFLAGS)

clean:
	rm -f run *.o
