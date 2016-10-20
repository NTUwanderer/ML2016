#ifndef UTIL_H
#define UTIL_H

#include <fstream>
#include <math.h>
void consumeCol(std::fstream* finp, int num) {
	char c;
	while (num > 0) {
		finp->get(c);
		if (c == ',' || c == 10 || c == 13)
			--num;
	}
}

void consumeCommaAndNewLine(std::fstream* finp) {
	while (true) {
		int c = finp->peek();
		if (c == 44 || c == 10 || c == 13)
			finp->get();
		else
			break;
	}
}

double square(const double& num) {
	return num * num;
}

void readAnswer(std::string fileName, double* const answer, const int& numPros) {
	std::fstream fin;
	fin.open(fileName.c_str(), std::ios::in);
	for (int i = 0; i < numPros; ++i)
		fin >> answer[i];

	fin.close();
}

double calculateError(const double* const answer, const double* const my_estimate, const int& numPros) {
	double error = 0;
	for (int i = 0; i < numPros; ++i)
		error += square(answer[i] - my_estimate[i]);
	error = sqrt(error / numPros);
	return error;
}

#endif
