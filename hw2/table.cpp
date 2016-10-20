#include "table.h"
#include <stdio.h>
#include <math.h>

extern double square(const double& num);

Table::Table() {
	trains = new Train[numTrains];
}

Table::~Table() {
	delete[] trains;
}

void Table::read(const string& csvFile) {
	std::fstream fin;
	fin.open(csvFile.c_str(), std::ios::in);

	for (int i = 0; i < numTrains; ++i)
		trains[i].read(&fin);

	fin.close();
}

const Train& Table::operator[] (size_t i) const {
	return trains[i];
}

Train Table::operator[] (size_t i) {
	return trains[i];
}

void Table::logisticRegression(const double& eta, double& b, double* const w, const double& deltaStop) const {
	// TODO
}

