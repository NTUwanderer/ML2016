#ifndef TRAIN_H
#define TRAIN_H

#include <iostream>
#include <fstream>

using namespace std;

class Train {
public:
   Train();
   ~Train();

   const double operator[] (size_t i) const;
   double operator[] (size_t i);

   void read(fstream* finp);
   void print() const;

   void logisticFunc(const double& b, const double* const w) const;

   const static int numCols = 58;

private:
   double* data;
};

#endif
