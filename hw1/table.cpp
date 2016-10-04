#include "table.h"
#include <stdio.h>

Table::Table() {
	months = new Month[Table::numMon];
}

Table::~Table() {
	delete[] months;
}

void consumeLine(fstream* finp) {
	char cs[100];
	finp->getline(cs, 100, char(10));
}

void Table::read(const string& csvFile) {
	fstream fin;
	fin.open(csvFile.c_str(), ios::in);

	consumeLine(&fin);

	for (int i = 0; i < Table::numMon; ++i)
		months[i].read(&fin);
	months[5].print();
	fin.close();
}
