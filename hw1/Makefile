CFLAGS = -g -Wall

run: main.o table.o month.o
	g++ -o $@ main.o table.o month.o $(CFLAGS)

main.o: main.cpp table.h
	g++ -c main.cpp $(CFLAGS)

table.o: table.cpp table.h month.h
	g++ -c table.cpp $(CFLAGS)

month.o: month.cpp month.h
	g++ -c month.cpp $(CFLAGS)

clean:
	rm -f run *.o
