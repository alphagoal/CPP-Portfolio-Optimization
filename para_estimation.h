
#ifndef para_estimation_h
#define para_estimation_h

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
using namespace std;
using Matrix2D = vector<vector<long double> >;
using Cube     = vector<Matrix2D>;  


class ParameterEstimator {
public:
    // Constructor
    ParameterEstimator(int windowSize = 100);
    
    // Main functionality - Static utility functions
    static Matrix2D rollingMeans(const Matrix2D& returns, int windowSize = 100);
    static Cube rollingCovariance(const Matrix2D& returns, const Matrix2D& meanreturns, int windowSize = 100);
    static double string_to_double(const string& s);
    static void readData(long double** data, const string& fileName);
    
    // Template method for data transformation
    template <class T>
    static vector<vector<T> > transform_to_vec(T** data, int rows, int cols) {
        vector<vector<T> > data_vec(rows, vector<T>(cols));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                data_vec[i][j] = data[i][j];
            }
        }
        return data_vec;
    }

private:
    // Private member variable, if any
    int windowSize_; //placeholder
};   

#endif