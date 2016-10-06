#ifndef TABLE_H
#define TABLE_H

#include "month.h"

class Table {
public:
  	Table();
   	~Table();

   	void read(const string& csvFile);
   	const Month& operator[](size_t i) const;
   	Month& operator[](size_t i);

   	void linearRegression(const double& eta, double& b, double* const w, const double& deltaStop, const int lambda = 0) const;
   	void linearRegression(const double& eta, double& b, double* const w, const int& index, double* const z, const double& deltaStop, const int lambda) const;
   	void quadraticRegression(const double& eta, double& b, double* const w, double* const z, const double& deltaStop, const int lambda = 0) const;
   	
   	const static int numMon = 12;
private:
   Month* months;
};

#endif
