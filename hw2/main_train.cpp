#include <iostream>
#include "table.h"
#include "util.h"
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <cstdlib>

using std::cout;
using std::cin;
using std::endl;

void writeModel(char* outputModel_fileName, const double& b, const double* const w) {
	fstream fout;

	fout.open(outputModel_fileName, ios::out);
	fout << b << '\n';
	for (int i = 0; i < Table::numCols - 1; ++i)
		fout << w[i] << ' ';
	fout.close();
}

int main(int argc, char** argv) {
	if (argc < 3) {
		cout << "train <training_data> <output_model> [options]";
		exit(1);
	}
	const clock_t begin_time = clock();

	const int numPros = 600;
	char* trainingData_fileName = argv[1];
	char* outputModel_fileName = argv[2];

	Table table = Table();
	table.read(trainingData_fileName);

	double eta = 0.1;
	double b;
	double *w;
	double deltaStop = 0.0001;

	int	*my_estimate = new int[numPros];

	fstream fout;

	bool logistic = false;
	bool linear = false;
	bool neural_network = false;
	int layer = 1;
	
	for (int i = 3; i < argc; ++i) {
		if (strncmp(argv[i], "--logistic", 10) == 0) {
			logistic = true;
		} else if (strncmp(argv[i], "--linear", 8) == 0) {
			linear = true;
		} else if (strncmp(argv[i], "--neural_network", 15) == 0) {
			neural_network = true;
		} else if (strncmp(argv[i], "--layer", 7) == 0) {
			if (i < argc - 1) {
				istringstream ss(argv[i + 1]);
				int index;
				if (ss >> index) {
					++i;
					layer = index;
				}
			}
		}
	}
	w = new double[Table::numCols - 1];

	if (logistic) {
		b = 0;
		for (int i = 0; i < Table::numCols - 1; ++i)
			w[i] = 0;
		
		table.logisticRegression(eta, b, w, deltaStop);

		cout << "b: " << b << endl;
		for (int i = 0; i < Table::numCols - 1; ++i)
			cout << "w[" << i << "]: " << w[i] << endl;
		
		writeModel(outputModel_fileName, b, w);
	} else if (linear) {
		b = 0.5;
		for (int i = 0; i < Table::numCols - 1; ++i)
			w[i] = 0;

		deltaStop = 0.000001;
		
		table.linearRegression(eta, b, w, deltaStop);

		cout << "b: " << b << endl;
		for (int i = 0; i < Table::numCols - 1; ++i)
			cout << "w[" << i << "]: " << w[i] << endl;
		
		writeModel(outputModel_fileName, b, w);
	} else if (neural_network && layer > 0) {
		int* numOfNodes = new int[layer];
		for (int i = 0; i < layer; ++i) {
			if (i == 0)	numOfNodes[i] = 30;
			else		numOfNodes[i] = 10;
		}

		table.neuralNetworkRegression(layer, numOfNodes, eta, b, w, deltaStop, outputModel_fileName);

		delete[] numOfNodes;
	}

	delete[] w;
	delete[] my_estimate;

	cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << " seconds took.\n";
}
