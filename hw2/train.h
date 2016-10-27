#ifndef TRAIN_H
#define TRAIN_H

#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;

class Train {
public:
    Train();
    ~Train();

    const double operator[] (size_t i) const;
    double operator[] (size_t i);

    void read(fstream* finp);
    void print() const;

    double linear_z(const double& b, const double* const w) const;
    void update_z(double& z, const int& index, const double& prev_p, const double& p);

    double cross_entropy(const double& sigma) const;
    double error_square(const double& z) const;
    double gradient(const double& sigma, const int& index) const;

    const static int numCols = 58;

private:
    double* data;
};

#endif
