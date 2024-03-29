#include "table.h"
#include <stdio.h>
#include <vector>
#include <algorithm>

extern double square(const double& num);

extern double func_sigma(const double& z);

Table::Table() {
	trains = new Train[numTrains];
}

Table::~Table() {
	delete[] trains;
}

void Table::read(const string& csvFile) {
	std::fstream fin;
	fin.open(csvFile.c_str(), std::ios::in);

	for (int i = 0; i < numTrains; ++i)
		trains[i].read(&fin);

	fin.close();
}

const Train& Table::operator[] (size_t i) const {
	return trains[i];
}

Train Table::operator[] (size_t i) {
	return trains[i];
}

void Table::logisticRegression(const double& eta, double& b, double* const w, const double& deltaStop) const {
	double preError, error = 1, gradient_b = 0, G_b_t = 0;
	std::vector<int> order;
	for (int i = 0, length = numCols - 1; i < length; ++i)
		order.push_back(i);

	double 	*z = new double[numTrains],
			*gradient_w = new double[numCols - 1],
			*G_w_t = new double[numCols - 1];

	for (int i = 0, length = numCols - 1; i < length; ++i)
		G_w_t[i] = 0;

	int counter = 0, idle = 0;

	for (int i = 0; i < numTrains; ++i)
		z[i] = trains[i].linear_z(b, w);

	while (true) {
		// if (counter > 1000)
		std::random_shuffle(order.begin(), order.end());

		++counter;
		preError = error, error = 0;

		for (int i = 0; i < numTrains; ++i)
			z[i] = trains[i].linear_z(b, w);

		gradient_b = 0;
		for (int i = 0; i < numTrains; ++i)
			gradient_b += trains[i].gradient(func_sigma(z[i]), -1);
		gradient_b /= numTrains;
		// G_b_t /= 2;
		G_b_t += square(gradient_b);

		double prev_b = b;
		b -= eta * gradient_b / sqrt(G_b_t);
		for (int i = 0; i < numTrains; ++i)
			trains[i].update_z(z[i], -1, prev_b, b);

		for (int i = 0, length = order.size(); i < length; ++i) {
			int index = order[i];
			gradient_w[index] = 0;

			for (int j = 0; j < numTrains; ++j)
				gradient_w[index] += trains[j].gradient(func_sigma(z[j]), index);

			gradient_w[index] /= numTrains;
			// G_w_t[index] /= 2;
			G_w_t[index] += square(gradient_w[index]);

			double prev_w_index = w[index];
			if (G_w_t[index] != 0)
				w[index] -= eta * gradient_w[index] / sqrt(G_w_t[index]);
			else
				w[index] -= eta * gradient_w[index];
			for (int j = 0; j < numTrains; ++j) {
				trains[j].update_z(z[j], index, prev_w_index, w[index]);
			}
		}

		for (int i = 0; i < numTrains; ++i)
			error += trains[i].cross_entropy(func_sigma(z[i]));

		error /= numTrains;
		cout << counter << ": " << error << endl;

		if (preError - error < deltaStop)
			++idle;
		else
			idle = 0;

		if (counter == 100000 || idle == 5)
			break;
	}
}

void Table::linearRegression(const double& eta, double& b, double* const w, const double& deltaStop) const {
	double preError, error = 1, gradient_b = 0, G_b_t = 0;
	vector<int> order;
	for (int i = 0, length = numCols - 1; i < length; ++i)
		order.push_back(i);
	// random the order later...

	double 	*z = new double[numTrains],
			*gradient_w = new double[numCols - 1],
			*G_w_t = new double[numCols - 1];

	for (int i = 0, length = numCols - 1; i < length; ++i)
		G_w_t[i] = 0;

	int counter = 0, idle = 0;

	for (int i = 0; i < numTrains; ++i)
		z[i] = trains[i].linear_z(b, w);

	while (true) {
		std::random_shuffle(order.begin(), order.end());

		++counter;
		preError = error, error = 0;

		for (int i = 0; i < numTrains; ++i)
			z[i] = trains[i].linear_z(b, w);

		gradient_b = 0;
		for (int i = 0; i < numTrains; ++i)
			gradient_b += trains[i].gradient(z[i], -1);
		gradient_b /= numTrains;
		G_b_t += square(gradient_b);

		double prev_b = b;
		b -= eta * gradient_b / sqrt(G_b_t);
		for (int i = 0; i < numTrains; ++i)
			trains[i].update_z(z[i], -1, prev_b, b);

		for (int i = 0, length = numCols - 1; i < length; ++i) {
			int index = order[i];
			gradient_w[index] = 0;

			for (int j = 0; j < numTrains; ++j)
				gradient_w[i] += trains[j].gradient(z[j], index);

			gradient_w[index] /= numTrains;
			G_w_t[index] += square(gradient_w[index]);

			double prev_w_index = w[index];
			if (G_w_t[index] != 0)
				w[index] -= eta * gradient_w[index] / sqrt(G_w_t[index]);
			else
				w[index] -= eta * gradient_w[index];
			for (int j = 0; j < numTrains; ++j) {
				trains[j].update_z(z[j], index, prev_w_index, w[index]);
			}
		}

		for (int i = 0; i < numTrains; ++i)
			error += trains[i].error_square(z[i]);

		error /= numTrains;
		cout << counter << ": " << error << endl;

		if (preError - error < deltaStop)
			++idle;
		else
			idle = 0;

		if (counter == 100000 || idle == 3)
			break;
	}
}
