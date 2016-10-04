#include <iostream>
#include "table.h"

using std::cout;
using std::cin;
using std::endl;

int main() {
	cout << "test" << endl;
	Table table = Table();
	table.read("data/train.csv");
}
