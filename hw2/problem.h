#ifndef PROBLEM_H
#define PROBLEM_H

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

class Problem {
public:
   Problem();
   ~Problem();

   const double operator[] (size_t i) const;
   double operator[] (size_t i);

   void read(fstream* finp);
   void print() const;

   int logistic_estimate(const double& b, const double* const w) const;
   int linear_estimate(const double& b, const double* const w) const;

   int layer_logistic_estimate(const int& nodes, const double& b, const double* const w) const;
   
   void clear();
   void push_back(double pred);

   void updateLayerByFeatures(const double& b, const double* const w);
   void updateLayerByPrevLayer(const double& b, const double* const w);
   const static int numCols = 57;

private:
   double* data;
   vector<double> layerNodes;
   vector<double> next_layerNodes;
};

#endif
