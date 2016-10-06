#include "month.h"

extern void consumeCol(std::fstream*, int);

extern void consumeCommaAndNewLine(std::fstream*);

extern double square(const double&);

Month::Month() {
	data = new double*[Month::numRow];
	for (int i = 0; i < Month::numRow; ++i)
		data[i] = new double[Month::numCol];
}

Month::~Month() {
	for (int i = 0; i < Month::numRow; ++i)
		delete[] data[i];
	delete[] data;
}

const double* const Month::operator[] (size_t i) const {
	return data[i];
}

double* Month::operator[] (size_t i) {
	return data[i];
}

void Month::read(fstream* finp) {
	for (int day = 0; day < 20; ++day) {
		for (int i = 0; i < Month::numRow; ++i) {
			consumeCol(finp, 3);
			for (int j = 0; j < 24; ++j) {
				if (finp->peek() == 'N') {
					consumeCol(finp, 1);
					consumeCommaAndNewLine(finp);
					data[i][24 * day + j] = -1.0;
				} else {
					*finp >> data[i][24 * day + j];
					consumeCommaAndNewLine(finp);
				}
			}
		}
	}
}

void Month::print() const {
	for (int i = 0; i < Month::numRow; ++i) {
		for (int j = 456; j < 480; ++j) {
			if (data[i][j] == -1.0) {
				cout << "NR\t";
			} else {
				cout << data[i][j] << '\t';
			}
		}
		cout << endl;
	}
}

void Month::linearFunc(const double& b, const double* const w, double& error, double& gradient_b, double* const gradient_w) const {
	double delta, predict;
	for (int i = 9; i < Month::numCol; ++i) {
		predict = b;		
		for (int j = 0; j < 9; ++j)
			predict += w[j] * data[Month::pmIndex][i + j - 9];
		delta = data[Month::pmIndex][i] - predict;
		error += square(delta);
		gradient_b -= 2 * delta;
		for (int j = 0; j < 9; ++j)
			gradient_w[j] -= 2 * delta * data[Month::pmIndex][i + j - 9];
	}
}

void Month::linearFunc(const double& b, const double* const w, const int& index, const double* z, double& error, double& gradient_b, double* const gradient_w, double* const gradient_z) const {
	double delta, predict;
	for (int i = 9; i < Month::numCol; ++i) {
		predict = b;
		for (int j = 0; j < 9; ++j) {
			predict += w[j] * data[Month::pmIndex][i + j - 9];
			predict += z[j] * data[index][i + j - 9];
		}
		delta = data[Month::pmIndex][i] - predict;
		error += square(delta);
		gradient_b -= 2 * delta;
		for (int j = 0; j < 9; ++j) {
			gradient_w[j] -= 2 * delta * data[Month::pmIndex][i + j - 9];
			gradient_z[j] -= 2 * delta * data[index][i + j - 9];
		}
	}
}

void Month::quadraticFunc(const double& b, const double* const w, const double* const z, double& error, double& gradient_b, double* const gradient_w, double* const gradient_z) const {
	double delta, predict;
	for (int i = 9; i < Month::numCol; ++i) {
		predict = b;
		for (int j = 0; j < 9; ++j) {
			predict += w[j] * data[Month::pmIndex][i + j - 9];
			predict += z[j] * square(data[Month::pmIndex][i + j - 9]);
		}
		delta = data[Month::pmIndex][i] - predict;
		error += square(delta);
		gradient_b -= 2 * delta;
		for (int j = 0; j < 9; ++j) {
			gradient_w[j] -= 2 * delta * data[Month::pmIndex][i + j - 9];
			gradient_z[j] -= 2 * delta * square(data[Month::pmIndex][i + j - 9]);
		}
	}
}
