#include "table.h"
#include <stdio.h>
#include <math.h>

double square(const double& num) {
	return num * num;
}

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
	fin.close();
	months[5].print();
}

const Month& Table::operator[] (size_t i) const {
	return months[i];
}

Month& Table::operator[] (size_t i) {
	return months[i];
}

void Table::linearRegression(const double& eta, double& b, double* const w, const double& deltaStop) {
	double preError, error = 0, gradient_b;
	double* gradient_w = new double[9];
	double G_b_t = 0;
	double* G_w_t = new double[9];
	for (int i = 0; i < 9; ++i)
		G_w_t[i] = 0;
	
	int counter = 0, idle = 0;

	while (true) {
		++counter;
		preError = error, error = 0, gradient_b = 0;
		for (int i = 0; i < 9; ++i)
			gradient_w[i] = 0;

		for (int i = 0; i < Table::numMon; ++i)
			months[i].goodnessOfFunc(b, w, error, gradient_b, gradient_w);

		error /= (12 * 471);
		cout << counter << ": " << error << endl;
		G_b_t += square(gradient_b);
		b -= eta * gradient_b / sqrt(G_b_t);
		for (int i = 0; i < 9; ++i) {
			G_w_t[i] += square(gradient_w[i]);
			w[i] -= eta * gradient_w[i] / sqrt(G_w_t[i]);
		}

		if (preError - error < deltaStop)
			++idle;
		else
			idle = 0;

		if (counter == 10000 || idle == 3) {
			cout << "counter: " << counter << endl;
			break;
		}
	}

	delete[] gradient_w;
}

