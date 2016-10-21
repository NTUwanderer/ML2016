#ifndef PROBLEM_H
#define PROBLEM_H

#include <iostream>
#include <fstream>

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

   const static int numCols = 57;

private:
   double* data;
};

#endif
