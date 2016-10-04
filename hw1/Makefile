CFLAGS = -g -Wall

run: main.o table.o month.o problem.o
	g++ -o $@ main.o table.o month.o problem.o $(CFLAGS)

main.o: main.cpp table.h problem.h
	g++ -c main.cpp $(CFLAGS)

table.o: table.cpp table.h month.h
	g++ -c table.cpp $(CFLAGS)

month.o: month.cpp month.h util.h
	g++ -c month.cpp $(CFLAGS)

problem.o: problem.cpp problem.h util.h
	g++ -c problem.cpp $(CFLAGS)

clean:
	rm -f run *.o
