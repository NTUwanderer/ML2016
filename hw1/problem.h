#ifndef PROBLEM_H
#define PROBLEM_H

#include <iostream>
#include <fstream>

using namespace std;

class Problem {
public:
   Problem();
   ~Problem();

   // const double* const operator[] (size_t) const;
   // double* operator[] (size_t);

   void read(fstream*);
   void print() const;

   double estimate(const double&, const double* const);

   const static int numRow = 18;
   const static int numCol = 9;
   const static int pmIndex = 9;
private:
   double** data;
};

#endif
