#ifndef MONTH_H
#define MONTH_H

#include <iostream>
#include <fstream>

using namespace std;

class Month {
public:
   Month();
   ~Month();

   const double* const operator[] (size_t i) const;
   double* operator[] (size_t i);

   void read(fstream* finp);
   void print() const;

   void linearFunc(const double& b, const double* const w, double& error, double& gradient_b, double* const gradient_w) const;
   void linearFunc(const double& b, const double* const w, const int& index, const double* z, double& error, double& gradient_b, double* const gradient_w, double* const gradient_z) const;
   void quadraticFunc(const double& b, const double* const w, const double* const z, double& error, double& gradient_b, double* const gradient_w, double* const gradient_z) const;

   const static int numRow = 18;
   const static int numCol = 480;
   const static int pmIndex = 9;
private:
   double** data;
};

#endif
