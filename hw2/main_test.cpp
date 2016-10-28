#include <iostream>
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
	if (argc < 4) {
		cout << "train <model_name> <testing_data> <prediction.csv> [options]";
		exit(1);
	}
	const clock_t begin_time = clock();

	const int numPros = 600;
	char* modelName_fileName = argv[1];
	char* testingData_fileName = argv[2];
	char* prediction_fileName = argv[3];

	double b;
	double *w = new double[Problem::numCols];;

	int	*my_estimate = new int[numPros];

	fstream fin_model;
	fstream fin;
	fstream fout;

	Problem* problems = new Problem[numPros];
	fin.open(testingData_fileName, ios::in);

	for (int i = 0; i < numPros; ++i)
		problems[i].read(&fin);
	fin.close();

	// for (int i = 0; i < numPros; ++i)
	// 	problems[i].print();
	// exit(1);

	bool logistic = false;
	bool linear = false;
	bool neural_network = false;
	
	string outputFile = "data/logistic_regression.csv";
	for (int i = 1; i < argc; ++i) {
		if (strncmp(argv[i], "--logistic", 10) == 0) {
			logistic = true;
		} else if (strncmp(argv[i], "--linear", 8) == 0) {
			linear = true;
		} else if (strncmp(argv[i], "--neural_network", 15) == 0) {
			neural_network = true;
		}
	}
	
	if (logistic) {
		fin_model.open(modelName_fileName, ios::in);
		fin_model >> b;
		for (int i = 0; i < Problem::numCols; ++i)
			fin_model >> w[i];
		fin_model.close();

		fout.open(prediction_fileName, ios::out);
		fout << "id,label\n";

		for (int i = 0; i < numPros; ++i) {
			my_estimate[i] = problems[i].logistic_estimate(b, w);
			std::ostringstream stm;
	        stm << i + 1;
			fout << stm.str() << ',' << my_estimate[i] << '\n';
		}
		fout.close();
	} else if (linear) {
		fin_model.open(modelName_fileName, ios::in);
		fin_model >> b;
		for (int i = 0; i < Problem::numCols; ++i)
			fin_model >> w[i];
		fin_model.close();

		fout.open(prediction_fileName, ios::out);
		fout << "id,label\n";

		for (int i = 0; i < numPros; ++i) {
			my_estimate[i] = problems[i].linear_estimate(b, w);
			std::ostringstream stm;
	        stm << i + 1;
			fout << stm.str() << ',' << my_estimate[i] << '\n';
		}
		fout.close();
	} else if (neural_network) {
		int layer;
		double b = 0;
		double* w;
		fin_model.open(modelName_fileName, ios::in);

		fin_model >> layer;
		int* numOfNodes = new int[layer];
		for (int i = 0; i < layer; ++i)
			fin_model >> numOfNodes[i];
		fin_model.close();

		for (int layerIndex = 0; layerIndex < layer; ++layerIndex) {
			int nodes;

			if (layerIndex == 0)	{
				nodes = Problem::numCols;
				const int times = numOfNodes[layerIndex];
				for (int i = 0; i < times; ++i) {
					w = new double[Problem::numCols];
					fin_model >> b;
					for (int j = 0; j < nodes; ++j)
						fin_model >> w[j];
					for (int j = 0; j < numPros; ++j) {
						problems[j].updateLayerByFeatures(b, w);
					}
				}
			} else if (layerIndex == 0)	{
				nodes = numOfNodes[layerIndex - 1];
				const int times = numOfNodes[layerIndex];
				for (int i = 0; i < times; ++i) {
					w = new double[Problem::numCols];
					fin_model >> b;
					for (int j = 0; j < nodes; ++j)
						fin_model >> w[j];
					for (int j = 0; j < numPros; ++j) {
						problems[j].updateLayerByPrevLayer(b, w);
					}
				}
			}
			for (int j = 0; j < numPros; ++j)
				problems[j].clear();

			delete[] w;
		}
		fout.open(prediction_fileName, ios::out);
		fout << "id,label\n";

		for (int i = 0; i < numPros; ++i) {
			my_estimate[i] = problems[i].layer_logistic_estimate(numOfNodes[layer - 1], b, w);
			std::ostringstream stm;
	        stm << i + 1;
			fout << stm.str() << ',' << my_estimate[i] << '\n';
		}
		fout.close();

		delete[] numOfNodes;
	}

	delete[] w;
	delete[] problems;
	delete[] my_estimate;

	cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << " seconds took.\n";
}
