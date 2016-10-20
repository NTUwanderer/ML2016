#include "problem.h"

extern void consumeCol(std::fstream*, int);

extern void consumeCommaAndNewLine(std::fstream*);

extern double square(const double&);

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

double Problem::logistic_estimate(const double& b, const double* const w) const {
	double ans = b;
	
	// TODO

	return ans;
}
