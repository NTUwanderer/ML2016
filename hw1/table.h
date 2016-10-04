#ifndef TABLE_H
#define TABLE_H

#include "month.h"

class Table {
public:
  	Table();
   	~Table();

   	void read(const string&);

   const static int numMon = 12;
private:
   Month* months;
};

#endif
