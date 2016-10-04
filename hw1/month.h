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
   double* operator[] (size_t i);

   void read(fstream*);
   void print() const;

   const static int numRow = 18;
   const static int numCol = 480;
private:
   double** data;
};

#endif
