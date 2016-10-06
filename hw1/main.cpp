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
	cout << "argc: " << argc << '\n';
	for (int i = 0; i < argc; ++i)
		cout << argv[i] << '\t';
	cout << '\n';
	Table table = Table();
	table.read("data/train.csv");

	double eta = 10;
	double b;
	double 	*w = new double[9],
			*z = new double[9];
	double deltaStop = 0.0001;

	fstream fin;
	fstream fout;
	const string testName = "data/test_X.csv";
	const int numPros = 240;
	Problem* problems = new Problem[numPros];
	fin.open(testName.c_str(), ios::in);

	for (int i = 0; i < numPros; ++i)
		problems[i].read(&fin);
	fin.close();

	if (argc == 1 || (argc == 2 && strncmp(argv[1], "--linear", 8) == 0)) {
		b = 10.0;	w[8] = 0.1;
		for (int i = 7; i >= 0; --i)
			w[i] = 0.8 * w[i + 1];

		table.linearRegression(eta, b, w, deltaStop, 0);

		cout << "b: " << b << endl;
		for (int i = 0; i < 9; ++i)
			cout << "w[" << i << "]: " << w[i] << endl;

		const string outName = "data/linear_regression.csv";
		fout.open(outName.c_str(), ios::out);
		fout << "id,value";

		for (int i = 0; i < numPros; ++i) {
			std::ostringstream stm ;
	        stm << i;
			fout << "\nid_" << stm.str() << ',' << problems[i].linear_estimate(b, w);
		}
		fout.close();
	} else if (argc == 3 && strncmp(argv[1], "--linear", 8) == 0) {
		cout << "-linear two \n";
		istringstream ss(argv[2]);
		int index;
		if (!(ss >> index))
		    cerr << "Invalid number " << argv[2] << '\n';
		b = 10.0, w[8] = 0.1, z[8] = 0;
		for (int i = 7; i >= 0; --i) {
			w[i] = 0.8 * w[i + 1];
			z[i] = 0.8 * z[i + 1];
		}

		table.linearRegression(eta, b, w, index, z, deltaStop, 0);

		cout << "b: " << b << endl;
		for (int i = 0; i < 9; ++i)
			cout << "w[" << i << "]: " << w[i] << endl;
		for (int i = 0; i < 9; ++i)
			cout << "z[" << i << "]: " << z[i] << endl;

		const string outName = "data/linear_regression_two.csv";
		fout.open(outName.c_str(), ios::out);
		fout << "id,value";

		for (int i = 0; i < numPros; ++i) {
			std::ostringstream stm ;
	        stm << i;
			fout << "\nid_" << stm.str() << ',' << problems[i].linear_estimate(b, w, index, z);
		}
		fout.close();
	} else if (argc == 2 && strncmp(argv[1], "--quadratic", 11) == 0) {
		// b = 10.0;	w[8] = 0.5, z[8] = 0.025;
		// for (int i = 7; i >= 0; --i) {
		// 	w[i] = 0.8 * w[i + 1];
		// 	z[i] = 0.8 * z[i + 1];
		// }

		b = 5.26203;
		w[0]= -0.0532044;
		w[1]= -0.0268756;
		z[1]= 0.000342083;
		w[2]= 0.0244257;
		z[2]= 0.00187016;
		w[3]= -0.068668;
		z[3]= -0.00173883;
		w[4]= 0.0149802;
		z[4]= -0.000401798;
		w[5]= 0.134812;
		z[5]= 0.00410514;
		w[6]= -0.231021;
		z[6]= -0.00414721;
		w[7]= 0.0957563;
		z[7]= -0.000770882;
		w[8]= 0.77836;
		z[8]= 0.00375687;

		// b = -2.35209;
		// w[0]= 0.11167;
		// z[0]= -0.00184616;
		// w[1]= -0.00918157;
		// z[1]= -5.0043e-05;
		// w[2]= 0.0933809;
		// z[2]= 0.00127349;
		// w[3]= -0.117595;
		// z[3]= -0.00134711;
		// w[4]= 0.0261888;
		// z[4]= -0.000821336;
		// w[5]= 0.367821;
		// z[5]= 0.00173437;
		// w[6]= -0.389562;
		// z[6]= -0.00239221;
		// w[7]= 0.0264444;
		// z[7]= -0.000192097;
		// w[8]= 1.09756;
		// z[8]= 0.000075621;


		table.quadraticRegression(eta, b, w, z, deltaStop, 0);

		cout << "b: " << b << endl;
		for (int i = 0; i < 9; ++i)
			cout << "w[" << i << "]: " << w[i] << endl;
		for (int i = 0; i < 9; ++i)
			cout << "z[" << i << "]: " << z[i] << endl;

		const string outName = "data/quadratic_regression.csv";
		fout.open(outName.c_str(), ios::out);
		fout << "id,value";

		for (int i = 0; i < numPros; ++i) {
			std::ostringstream stm ;
	        stm << i;
			fout << "\nid_" << stm.str() << ',' << problems[i].quadratic_estimate(b, w, z);
		}
		fout.close();
	}
	delete[] w;
	delete[] z;
	delete[] problems;
}
