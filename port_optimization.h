#ifndef port_optimization_h
#define port_optimization_h

#include <iostream>
#include <cmath>
#include <vector>

using namespace std;
using Vec = vector <long double>;
using Matrix2D = vector<vector<long double> >;
using Cube     = vector<Matrix2D>;  

class PortfolioOptimizer {
public:
    // Main public functions
    static Matrix2D KKTMatrix_Q(const Matrix2D &Sigma, const Vec &meanreturn);
    static Vec RHS_b(int N, long double targetreturn);
    static Vec solveKKT_x(const Matrix2D &Q, const Vec &b);
    static void writeMatrixToCSV(const Matrix2D& matrix, const string& filename);
    static void writeVectorToCSV(const Vec& vector, const string& filename);

private:
    // Private helper functions
    static long double dot(const Vec &a, const Vec &b) {
        long double sum = 0.0L;
        for (int i = 0; i < a.size(); ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    static Vec MatM_Vec(const Matrix2D &A, const Vec &x) {
        int n = A.size();
        int m = x.size();
        Vec b(n, 0.0L);
        for (int i = 0; i < n; ++i) {
            long double sum = 0.0L;
            for (int j = 0; j < m; ++j) {
                sum += A[i][j] * x[j];
            }
            b[i] = sum;
        }
        return b;
    }
    
    static void alpha_x_plus_y(long double alpha, const Vec &x, Vec &y) {
        for (int i = 0; i < x.size(); ++i) {
            y[i] += alpha * x[i];
        }
    }
};


#endif
