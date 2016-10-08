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
	const int numPros = 240;
	const string trainingData_fileName = "data/train.csv";
	const string testingData_fileName = "data/test_X.csv";

	const int numCols = Table::numCols; // 18

	Table table = Table();
	table.read(trainingData_fileName);

	double eta = 1;
	double b;
	double *w, *z;
	double deltaStop = 0.00001;

	double	*my_estimate	= new double[numPros],
			*sum 			= new double[numCols],
			*scale 			= new double[numCols];

	fstream fin;
	fstream fout;
	
	Problem* problems = new Problem[numPros];
	fin.open(testingData_fileName.c_str(), ios::in);

	for (int i = 0; i < numPros; ++i)
		problems[i].read(&fin);
	fin.close();

	bool linear = false;
	bool quadratic = false;
	bool featureScaling = false;
	bool all = false;
	bool prod_linear = false;
	bool prod_best = false;
	int columnIndex = -1;
	int lenOfTrain = 9;
	int lambda = 0;
	string outputFile = "data/linear_regression.csv";
	for (int i = 1; i < argc; ++i) {
		if (strncmp(argv[i], "--linear", 8) == 0) {
			linear = true;
			if (i < argc - 1) {
				istringstream ss(argv[i + 1]);
				int index;
				if (ss >> index) {
					++i;
					columnIndex = index;
				}
			}
		} else if (strncmp(argv[i], "--quadratic", 11) == 0) {
			quadratic = true;
		} else if (strncmp(argv[i], "--featureScaling", 16) == 0) {
			featureScaling = true;
		} else if (strncmp(argv[i], "--len", 5) == 0) {
			if (i < argc - 1) {
				istringstream ss(argv[i + 1]);
				int index;
				if (ss >> index) {
					++i;
					lenOfTrain = index;
				}
			}
		} else if (strncmp(argv[i], "--all", 5) == 0) {
			all = true;
		} else if (strncmp(argv[i], "--regularization", 16) == 0) {
			lambda = 1;
		} else if (strncmp(argv[i], "--prod_linear", 13) == 0) {
			prod_linear = true;
			outputFile = "linear_regression.csv";
		} else if (strncmp(argv[i], "--prod_best", 11) == 0) {
			prod_best = true;
			outputFile = "kaggle_best.csv";
		}
	}
	w = new double[lenOfTrain];
	z = new double[lenOfTrain];

	if (featureScaling) {
		for (int i = 0; i < numCols; ++i) {
			sum[i] = 0;
			for (int j = 0; j < 12; ++j)
				for (int k = 0; k < 480; ++k)
					sum[i] += table[j][i][k];
		}

		for (int i = 0; i < numCols; ++i) {
			if (i != Month::pmIndex) {
				scale[i] = abs(sum[Month::pmIndex] / sum[i]);
				for (int j = 0; j < 12; ++j)
					for (int k = 0; k < 480; ++k)
						table[j][i][k] *= scale[i];

				for (int j = 0; j < numPros; ++j)
					for (int k = 0; k < 9; ++k)
						problems[j][i][k] *= scale[i];
			}
		}
	}
	if ((linear && all) || prod_best) {
		b = -10.0;
		double** weight = new double*[numCols];
		for (int i = 0; i < numCols; ++i) {
			weight[i] = new double[lenOfTrain];
			if (i == Month::pmIndex) 	weight[i][lenOfTrain - 1] = 1;
			else 						weight[i][lenOfTrain - 1] = 0;
			for (int j = lenOfTrain - 2; j >= 0; --j)
				weight[i][j] = 0.1 * weight[i][j + 1];
		}

		table.linearRegression(lenOfTrain, eta, b, weight, deltaStop, lambda);

		cout << "b: " << b << endl;
		for (int i = 0; i < numCols; ++i) {
			printf("weight[%d]: ", i);
			for (int j = 0; j < lenOfTrain; ++j)
				printf("%f\t", weight[i][j]);
			printf("\n");
		}

		const string outName = (prod_best) ? outputFile : "data/linear_regression_all.csv";
		fout.open(outName.c_str(), ios::out);
		fout << "id,value";

		for (int i = 0; i < numPros; ++i) {
			my_estimate[i] = problems[i].linear_estimate(lenOfTrain, b, weight);
			std::ostringstream stm;
	        stm << i;
			fout << "\nid_" << stm.str() << ',' << my_estimate[i];
		}
		fout.close();

		for (int i = 0; i < numCols; ++i)
			delete[] weight[i];
		delete[] weight;
	} else if (linear && columnIndex == -1) {
		b = -10.0;	w[lenOfTrain - 1] = 1;
		for (int i = lenOfTrain - 2; i >= 0; --i)
			w[i] = 0.5 * w[i + 1];

		table.linearRegression(lenOfTrain, eta, b, w, deltaStop, lambda);

		cout << "b: " << b << endl;
		for (int i = 0; i < lenOfTrain; ++i)
			cout << "w[" << i << "]: " << w[i] << endl;

		const string outName = "data/linear_regression.csv";
		fout.open(outName.c_str(), ios::out);
		fout << "id,value";

		for (int i = 0; i < numPros; ++i) {
			my_estimate[i] = problems[i].linear_estimate(lenOfTrain, b, w);
			std::ostringstream stm;
	        stm << i;
			fout << "\nid_" << stm.str() << ',' << my_estimate[i];
		}
		fout.close();
	} else if (linear && columnIndex >= 0 && columnIndex < numCols) {
		cout << "-linear two \n";
		istringstream ss(argv[2]);
		int index;
		if (!(ss >> index))
		    cerr << "Invalid number " << argv[2] << '\n';
		b = -10.0, w[lenOfTrain - 1] = 1.0, z[lenOfTrain - 1] = 1.0;
		for (int i = lenOfTrain - 2; i >= 0; --i) {
			w[i] = 0.5 * w[i + 1];
			z[i] = 0.5 * z[i + 1];
		}

		table.linearRegression(lenOfTrain, eta, b, w, index, z, deltaStop, lambda);

		cout << "b: " << b << endl;
		for (int i = 0; i < lenOfTrain; ++i)
			cout << "w[" << i << "]: " << w[i] << endl;
		for (int i = 0; i < lenOfTrain; ++i)
			cout << "z[" << i << "]: " << z[i] << endl;

		const string outName = "data/linear_regression_two.csv";
		fout.open(outName.c_str(), ios::out);
		fout << "id,value";

		for (int i = 0; i < numPros; ++i) {
			my_estimate[i] = problems[i].linear_estimate(lenOfTrain, b, w, index, z);
			std::ostringstream stm;
	        stm << i;
			fout << "\nid_" << stm.str() << ',' << my_estimate[i];
		}
		fout.close();
	} else if (quadratic || prod_linear) {
		b = 10.0;	w[lenOfTrain - 1] = 0.5, z[lenOfTrain - 1] = 0.025;
		for (int i = lenOfTrain - 2; i >= 0; --i) {
			w[i] = 0.8 * w[i + 1];
			z[i] = 0.8 * z[i + 1];
		}

		table.quadraticRegression(lenOfTrain, eta, b, w, z, deltaStop, lambda);

		cout << "b: " << b << endl;
		for (int i = 0; i < lenOfTrain; ++i)
			cout << "w[" << i << "]: " << w[i] << endl;
		for (int i = 0; i < lenOfTrain; ++i)
			cout << "z[" << i << "]: " << z[i] << endl;

		const string outName = (prod_linear) ? outputFile : "data/quadratic_regression_all.csv";
		fout.open(outName.c_str(), ios::out);
		fout << "id,value";

		for (int i = 0; i < numPros; ++i) {
			my_estimate[i] = problems[i].quadratic_estimate(lenOfTrain, b, w, z);
			std::ostringstream stm ;
	        stm << i;
			fout << "\nid_" << stm.str() << ',' << my_estimate[i];
		}
		fout.close();
	}

	delete[] w;
	delete[] z;
	delete[] problems;
	delete[] my_estimate;
	delete[] sum;
	delete[] scale;
}
