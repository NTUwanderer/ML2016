#include <iostream>
#include "table.h"
#include "problem.h"
#include "util.h"
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <cstdlib>

using std::cout;
using std::cin;
using std::endl;

int main(int argc, char** argv) {
	const int numPros = 600;
	const string trainingData_fileName = "data/spam_train.csv";
	const string testingData_fileName = "data/spam_test.csv";

	Table table = Table();
	table.read(trainingData_fileName);

	double eta = 1;
	double b;
	double *w;
	double deltaStop = 0.00001;

	double	*my_estimate = new double[numPros];

	fstream fin;
	fstream fout;
	
	Problem* problems = new Problem[numPros];
	fin.open(testingData_fileName.c_str(), ios::in);

	for (int i = 0; i < numPros; ++i)
		problems[i].read(&fin);
	fin.close();

	bool logistic = false;
	
	string outputFile = "data/logistic_regression.csv";
	for (int i = 1; i < argc; ++i) {
		if (strncmp(argv[i], "--logistic", 10) == 0) {
			logistic = true;
		}
	}
	w = new double[Problem::numCols];

	
	if (logistic) {
		string outName = outputFile;
		
		b = 0.5;
		for (int i = 0; i < Problem::numCols; ++i)
			w[i] = 0;
		

		table.logisticRegression(eta, b, w, deltaStop);

		cout << "b: " << b << endl;
		for (int i = 0; i < Problem::numCols; ++i)
			cout << "w[" << i << "]: " << w[i] << endl;
		
		fout.open(outName.c_str(), ios::out);
		fout << "id,label\n";

		for (int i = 0; i < numPros; ++i) {
			my_estimate[i] = problems[i].logistic_estimate(b, w);
			std::ostringstream stm;
	        stm << i;
			fout << stm.str() << ',' << my_estimate[i] << '\n';
		}
		fout.close();
	}

	delete[] w;
	delete[] problems;
	delete[] my_estimate;
}
