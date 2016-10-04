#include "month.h"

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

void consumeCol(fstream* finp, int num) {
	char c;
	while (num > 0) {
		finp->get(c);
		if (c == ',' || c == 10 || c == 13)
			--num;
	}
}

void consumeCommaAndNewLine(fstream* finp) {
	while (true) {
		int c = finp->peek();
		if (c == 44 || c == 10 || c == 13)
			finp->get();
		else
			break;
	}
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
