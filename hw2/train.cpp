#include "train.h"

extern void consumeCol(std::fstream*, int);

extern void consumeCommaAndNewLine(std::fstream*);

extern double square(const double& num);

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

double Train::linear_z(const double& b, const double* const w) const {
	double z = b;
	for (int i = 0, length = numCols - 1; i < length; ++i)
		z += w[i] * data[i];

	return z;
}

void Train::update_z(double& z, const int& index, const double& prev_p, const double& p) {
	double temp_data = 1.0;
	if (index != -1)
		temp_data = data[index];

	z += temp_data * (p - prev_p);
}

double Train::gradient(const double& sigma, const int& index) const {
	if (index == -1)
		return -(data[numCols - 1] - sigma);
	return -(data[numCols - 1] - sigma) * data[index];
}

double Train::cross_entropy(const double& sigma) const {
	if (data[numCols - 1] == 1)
		return -log(sigma);
	else if (data[numCols - 1] == 0)
		return -log(1 - sigma);
	else {
		printf("Wrong label...");
		std::cin.get();
		return 0;
	}
}

double Train::error_square(const double& z) const {
	return square(z - data[numCols - 1]);
}
