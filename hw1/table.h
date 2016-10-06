#ifndef TABLE_H
#define TABLE_H

#include "month.h"

class Table {
public:
  	Table();
   	~Table();

   	void read(const string&);
   	const Month& operator[](size_t) const;
   	Month& operator[](size_t);

   	void linearRegression(const double&, double&, double* const, const double&, const int = 0);
   	void quadraticRegression(const double&, double&, double* const, double* const, const double&, const int = 0);
   	
   	const static int numMon = 12;
private:
   Month* months;
};

#endif
