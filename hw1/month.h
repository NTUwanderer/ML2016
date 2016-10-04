#ifndef MONTH_H
#define MONTH_H

#include <iostream>
#include <fstream>

using namespace std;

class Month {
public:
   Month();
   ~Month();

   const double* const operator[] (size_t) const;
   double* operator[] (size_t);

   void read(fstream*);
   void print() const;

   void goodnessOfFunc(const double&, const double* const, double&, double&, double* const) const;

   const static int numRow = 18;
   const static int numCol = 480;
   const static int pmIndex = 9;
private:
   double** data;
};

#endif
