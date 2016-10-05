#include <iostream>
#include "table.h"
#include "problem.h"
#include <string>
#include <sstream>

using std::cout;
using std::cin;
using std::endl;

int main() {
	cout << "test" << endl;
	Table table = Table();
	table.read("data/train.csv");

	double eta = 0.1;
	double b = 10.0;
	double* w = new double[9];
	double deltaStop = 0.0001;

	w[8] = 0.1;
	for (int i = 7; i >= 0; --i)
		w[i] = 0.8 * w[i + 1];

	table.linearRegression(eta, b, w, deltaStop, 0);

	cout << "b: " << b << endl;
	for (int i = 0; i < 9; ++i)
		cout << "w[" << i << "]: " << w[i] << endl;

	fstream fin;
	const string testName = "data/test_X.csv";
	const int numPros = 240;
	Problem* problems = new Problem[numPros];
	fin.open(testName.c_str(), ios::in);

	for (int i = 0; i < numPros; ++i)
		problems[i].read(&fin);
	fin.close();
	
	fstream fout;
	const string outName = "data/output.csv";
	fout.open(outName.c_str(), ios::out);
	fout << "id,value";

	for (int i = 0; i < numPros; ++i) {
		std::ostringstream stm ;
        stm << i;
		fout << "\nid_" << stm.str() << ',' << problems[i].estimate(b, w);
	}

	delete[] w;
	delete[] problems;
}
