CFLAGS = -Wall -g

all: run main_train main_test

run: main.o table.o train.o problem.o
	g++ -o $@ main.o table.o train.o problem.o $(CFLAGS) -o run

main_train: main_train.o table.o train.o
	g++ -o $@ main_train.o table.o train.o $(CFLAGS) -o main_train

main_test: main_test.o problem.o
	g++ -o $@ main_test.o problem.o $(CFLAGS) -o main_test

main.o: main.cpp
	g++ -c main.cpp $(CFLAGS)

main_train.o: main_train.cpp
	g++ -c main_train.cpp $(CFLAGS)

main_test.o: main_test.cpp
	g++ -c main_test.cpp $(CFLAGS)

table.o: table.cpp
	g++ -c table.cpp $(CFLAGS)

train.o: train.cpp
	g++ -c train.cpp $(CFLAGS)

problem.o: problem.cpp
	g++ -c problem.cpp $(CFLAGS)

# util.o: util.cpp
# 	g++ -c util.cpp $(CFLAGS)

clean:
	rm -f run main_train main_test *.o
