#ifndef UTIL_H
#define UTIL_H

#include <fstream>

extern void consumeCol(std::fstream* finp, int num) {
	char c;
	while (num > 0) {
		finp->get(c);
		if (c == ',' || c == 10 || c == 13)
			--num;
	}
}

extern void consumeCommaAndNewLine(std::fstream* finp) {
	while (true) {
		int c = finp->peek();
		if (c == 44 || c == 10 || c == 13)
			finp->get();
		else
			break;
	}
}

#endif
