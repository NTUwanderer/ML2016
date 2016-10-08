#include "problem.h"

extern void consumeCol(std::fstream*, int);

extern void consumeCommaAndNewLine(std::fstream*);

extern double square(const double&);

Problem::Problem() {
	data = new double*[Problem::numRow];
	for (int i = 0; i < Problem::numRow; ++i)
		data[i] = new double[Problem::numCol];
}

Problem::~Problem() {
	for (int i = 0; i < Problem::numRow; ++i)
		delete[] data[i];
	delete[] data;
}

const double* const Problem::operator[] (size_t i) const {
	return data[i];
}

double* Problem::operator[] (size_t i) {
	return data[i];
}

void Problem::read(fstream* finp) {
	for (int i = 0; i < Problem::numRow; ++i) {
		consumeCol(finp, 2);
		for (int j = 0; j < Problem::numCol; ++j) {
			if (finp->peek() == 'N') {
				consumeCol(finp, 1);
				consumeCommaAndNewLine(finp);
				data[i][j] = -1.0;
			} else {
				*finp >> data[i][j];
				consumeCommaAndNewLine(finp);
			}
		}
	}
}

void Problem::print() const {
	for (int i = 0; i < Problem::numRow; ++i) {
		for (int j = 0; j < Problem::numCol; ++j) {
			if (data[i][j] == -1.0) {
				cout << "NR\t";
			} else {
				cout << data[i][j] << '\t';
			}
		}
		cout << endl;
	}
}

double Problem::linear_estimate(const int& lenOfTrain, const double& b, const double* const w) const {
	double ans = b;
	for (int i = 0; i < lenOfTrain; ++i)
		ans += w[i] * data[Problem::pmIndex][i + 9 - lenOfTrain];

	return ans;
}

double Problem::linear_estimate(const int& lenOfTrain, const double& b, const double* const w, const int& index, const double* const z) const {
	double ans = b;
	for (int i = 0; i < lenOfTrain; ++i) {
		ans += w[i] * data[Problem::pmIndex][i + 9 - lenOfTrain];
		ans += z[i] * data[index][i + 9 - lenOfTrain];
	}

	return ans;
}

double Problem::linear_estimate(const int& lenOfTrain, const double& b, const double*const *const w) const {
	double ans = b;
	for (int i = 0; i < numRow; ++i)
		for (int j = 0; j < lenOfTrain; ++j)
			ans += w[i][j] * data[i][j + 9 - lenOfTrain];

	return ans;
}

double Problem::quadratic_estimate(const int& lenOfTrain, const double& b, const double* const w, const double* const z) const {
	double ans = b;
	for (int i = 0; i < lenOfTrain; ++i) 
		ans += (w[i] * data[Problem::pmIndex][i + 9 - lenOfTrain] + z[i] * square(data[Problem::pmIndex][i + 9 - lenOfTrain]));

	return ans;
}
