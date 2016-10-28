#ifndef TABLE_H
#define TABLE_H

#include "train.h"

using std::string;

class Table {
public:
  	Table();
	~Table();

	void read(const string& csvFile);
	const Train& operator[](size_t i) const;
	Train operator[](size_t i);

	void logisticRegression(const double& eta, double& b, double* const w, const double& deltaStop) const;
	void linearRegression(const double& eta, double& b, double* const w, const double& deltaStop) const;

	void layer_logisticRegression(const int& nodes, const double& eta, double& b, double* const w, const double& deltaStop) const;
	void neuralNetworkRegression(const int& layer, const int* const numOfNodes, const double& eta, double& b, double* const w, const double& deltaStop, const char* const outputModel_fileName) const;
	const static int numCols = Train::numCols;
	const static int numTrains = 4001;
private:
   Train* trains;
};

#endif
