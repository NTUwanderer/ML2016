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
}

const Month& Table::operator[] (size_t i) const {
	return months[i];
}

Month& Table::operator[] (size_t i) {
	return months[i];
}

void Table::linearRegression(const double& eta, double& b, double* const w, const double& deltaStop, const int lambda) {
	double preError, error = 200, gradient_b, regularization, G_b_t = 0;
	double *gradient_w = new double[9], *G_w_t = new double[9];
	for (int i = 0; i < 9; ++i)
		G_w_t[i] = 0;
	
	int counter = 0, idle = 0;
	while (true) {
		++counter;
		preError = error, error = 0, gradient_b = 0, regularization = 0;
		for (int i = 0; i < 9; ++i)
			gradient_w[i] = 0;

		for (int i = 0; i < Table::numMon; ++i)
			months[i].goodnessOfFunc(b, w, error, gradient_b, gradient_w);

		error /= (12 * 471);
		cout << counter << ": " << error << endl;
		G_b_t += square(gradient_b);
		b -= eta * gradient_b / sqrt(G_b_t);
		// b -= eta * gradient_b;
		for (int i = 0; i < 9; ++i)
			regularization += lambda * square(w[i]);
		// error += regularization;
		for (int i = 0; i < 9; ++i) {
			gradient_w[i] += 2 * lambda * w[i];
			G_w_t[i] += square(gradient_w[i]);
			w[i] -= eta * gradient_w[i] / sqrt(G_w_t[i]);
			// w[i] -= eta * gradient_w[i];
		}

		if (preError - error < deltaStop)
			++idle;
		else
			idle = 0;

		if (counter == 10000 || idle == 3)
			break;
	}

	delete[] gradient_w;
}

