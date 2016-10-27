#include "problem.h"

extern void consumeCol(std::fstream*, int);

extern void consumeCommaAndNewLine(std::fstream*);

extern double func_sigma(const double& z);

Problem::Problem() {
	data = new double[numCols];
}

Problem::~Problem() {
	delete[] data;
}

const double Problem::operator[] (size_t i) const {
	return data[i]; 
}

double Problem::operator[] (size_t i) {
	return data[i];
}

void Problem::read(fstream* finp) {
	consumeCol(finp, 1);
	for (int i = 0; i < numCols; ++i) {
		*finp >> data[i];
		consumeCommaAndNewLine(finp);
	}
}

void Problem::print() const {
	for (int i = 0; i < numCols; ++i)
		cout << data[i] << '\t';
	cout << endl;
}

int Problem::logistic_estimate(const double& b, const double* const w) const {
	int ans;
	double z = b;
	for (int i = 0, length = numCols; i < length; ++i)
		z += w[i] * data[i];
	
	if (z >= 0)	ans = 1;
	else		ans = 0;

	return ans;
}

int Problem::linear_estimate(const double& b, const double* const w) const {
	int ans;
	double z = b;
	for (int i = 0, length = numCols; i < length; ++i)
		z += w[i] * data[i];
	
	if (z >= 0.5)	ans = 1;
	else			ans = 0;

	return ans;
}
