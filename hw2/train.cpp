#include "train.h"

extern void consumeCol(std::fstream*, int);

extern void consumeCommaAndNewLine(std::fstream*);

extern double square(const double&);

Train::Train() {
	data = new double[numCols];
}

Train::~Train() {
	delete[] data;
}

const double Train::operator[] (size_t i) const {
	return data[i]; 
}

double Train::operator[] (size_t i) {
	return data[i];
}

void Train::read(fstream* finp) {
	consumeCol(finp, 1);
	for (int i = 0; i < numCols; ++i) {
		*finp >> data[i];
		consumeCommaAndNewLine(finp);
	}
}

void Train::print() const {
	for (int i = 0; i < numCols; ++i)
		cout << data[i] << '\t';
	cout << endl;
}

void Train::logisticFunc(const double& b, const double* const w) const {	
	// TODO
}
