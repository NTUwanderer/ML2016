CFLAGS = -g -Wall

run: main.o table.o month.o problem.o
	g++ -o $@ main.o table.o month.o problem.o $(CFLAGS)

main.o: main.cpp
	g++ -c main.cpp $(CFLAGS)

table.o: table.cpp
	g++ -c table.cpp $(CFLAGS)

month.o: month.cpp
	g++ -c month.cpp $(CFLAGS)

problem.o: problem.cpp
	g++ -c problem.cpp $(CFLAGS)

# util.o: util.cpp
# 	g++ -c util.cpp $(CFLAGS)

clean:
	rm -f run *.o
