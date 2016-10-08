#ifndef PROBLEM_H
#define PROBLEM_H

#include <iostream>
#include <fstream>

using namespace std;

class Problem {
public:
   Problem();
   ~Problem();

   const double* const operator[] (size_t i) const;
   double* operator[] (size_t i);

   void read(fstream* finp);
   void print() const;

   double linear_estimate(const int& lenOfTrain, const double& b, const double* const w) const;
   double linear_estimate(const int& lenOfTrain, const double& b, const double* const w, const int& index, const double* const z) const;
   double linear_estimate(const int& lenOfTrain, const double& b, const double*const *const w) const;
   double quadratic_estimate(const int& lenOfTrain, const double& b, const double* const w, const double* const z) const;

   const static int numRow = 18;
   const static int numCol = 9;
   const static int pmIndex = 9;
private:
   double** data;
};

#endif
