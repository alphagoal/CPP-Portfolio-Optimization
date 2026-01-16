#include <iostream>
#include <cmath>
#include <vector>
#include "para_estimation.h"
#include "port_optimization.h"
#include "csv.h"
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <numeric>  

using namespace std;
using Vec = vector <long double>;
using Matrix2D = vector<vector<long double> >;
using Cube     = vector<Matrix2D>;


Matrix2D PortfolioOptimizer::KKTMatrix_Q(const Matrix2D &Sigma, const Vec &meanreturn){
    int N = Sigma.size(); //find the number of assets
    int M = N+2; 

    cout << "Checkpoint step 0: Algorithm has entered the function ......" << endl;
    cout << "Start Step 1 ......" << endl;

    //Step 1: Create a Zero Matrix
    Matrix2D Q(M, Vec(M,0.0L));

    cout << "\nCheckpoint step 1: Zero Matrix is created ......" << endl;
    cout << "********* Start Step 2 ...... ********* " << endl;

    //Step 2: Fill up the non-zero blocks in the matrix

    cout << "Start Top Left Sigma ......" << endl;
    //Top left sigma
    for (int i = 0; i < N; ++i){
        for (int j=0; j<N; ++j){
            Q[i][j] = Sigma[i][j];
        }
    }
    cout << "Finish Top Left Sigma ......" << endl;
    cout << "\nStart Top Right (meanreturn and e) ......" << endl;

    //Top right: -meanreturn and -e
    for (int i = 0; i < N; ++i){
        Q[i][N] = -meanreturn[i];
        Q[i][N+1] = -1.0L;
    }
    cout << "\nFinish Top Right (meanreturn and e) ......" << endl;

    cout << "Start Bottom Left  Transpose of (meanreturn and e) ......" << endl;
    //Bottom left: transpose of -meanreturn and -e
    for (int j = 0; j < N; ++j){
        Q[N][j] = -meanreturn[j];
        Q[N+1][j] = -1.0L;
    }
    cout << "Finish Bottom Left  Transpose of (meanreturn and e) ......" << endl;

    //Bottom right 2x2 space are still zeros
    cout << "Ready to return Matrix2D 'Q' " << endl;
    return Q;
};


Vec PortfolioOptimizer::RHS_b(int N, long double targetreturn){
    Vec b(N+2, 0.0L);
    b[N] = -targetreturn;
    b[N+1] = -1.0L;
    return b;
};

//improved version
Vec PortfolioOptimizer::solveKKT_x(const Matrix2D &Q, const Vec &b) {
    cout << "Start processing improved solverKKT function......" << endl;
    
    int M = b.size();
    int N = M - 2;  // Number of assets
    long double targetReturn = -b[N]; 
    
    cout << "Matrix size: " << M << "x" << M << endl;
    cout << "Number of assets: " << N << endl;
    
    // Step 1: Create better initial guess
    Vec x(M, 0.0L);
    
    // Set initial portfolio weights to equal weights that sum to 1 for guessing
    for(int i = 0; i < N; i++) {
        x[i] = 1.0L / N;
    }
    
    // Set initial Lagrange multipliers to reasonable values
    x[N] = targetReturn;      // λ (return constraint)
    x[N+1] = 1.0L;     // μ (budget constraint)
    
    cout << "Initial guess created:" << endl;
    cout << "Portfolio weight sum: " << N * (1.0L / N) << " (should be 1.0)" << endl;
    cout << "Initial lambda: " << x[N] << endl;
    cout << "Initial mu: " << x[N+1] << endl;
    
    // Step 2: Calculate initial residual r = b - Q*x
    Vec Q_x = MatM_Vec(Q, x);
    Vec r(M);
    for(int i = 0; i < M; i++) {
        r[i] = b[i] - Q_x[i];
    }
    
    // Step 3: Initialize search direction
    Vec p = r;  // p0 = r0
    
    long double tolerance = 1e-12L;  // tighter tolerance is preferred //*********************** */
    long double rsold = dot(r, r);   // r^T * r
    
    cout << "Initial residual norm squared: " << rsold << endl;
    cout << "Tolerance: " << tolerance << endl;
    cout << "Start processing Conjugate Gradient Program......" << endl;

    // Step 4: Main CG iteration loop
    for (int k = 0; k < M && rsold > tolerance; ++k) {   //*********************** */
        cout << "Iteration " << k << "..................................................." << endl;
        cout << "Current residual norm: " << sqrt(rsold) << endl;
        
        // Compute Q*p
        Vec Qp = MatM_Vec(Q, p);
        
        // Compute p^T * Q * p
        long double pQp = dot(p, Qp);
        cout << "p^T * Q * p = " << pQp << endl;
        
        // Check for stopping condition breakdown or indefinite matrix
        if (abs(pQp) < 1e-14) {
            cout << "Warning: p^T * Q * p is nearly zero (" << pQp << "), stopping" << endl;
            break;
        }
        
        // Compute step size alpha = (r^T * r) / (p^T * A * p)
        long double alpha = rsold / pQp;
        cout << "Alpha = " << alpha << endl;
        
        // Check for reasonable alpha
        if (abs(alpha) > 1e6) {
            cout << "Warning: Alpha is very large (" << alpha << "), may be unstable" << endl;
        }
        
        // Update solution: x = x + alpha * p
        alpha_x_plus_y(alpha, p, x);
        
        // Update residual: r = r - alpha * A*p
        alpha_x_plus_y(-alpha, Qp, r);
        
        // Compute new residual norm squared
        long double rsnew = dot(r, r);
        cout << "New residual norm: " << sqrt(rsnew) << endl;
        
        // Check convergence
        if (sqrt(rsnew) < tolerance) {
            cout << "Converged at iteration " << k << " with residual " << sqrt(rsnew) << endl;
            break;
        }
        
        // Check for stagnation
        if (rsnew > 0.99 * rsold && k > 10) {
            cout << "Warning: Slow convergence detected at iteration " << k << endl;
        }
        
        // Compute beta = (r_new^T * r_new) / (r_old^T * r_old)
        long double beta = rsnew / rsold;
        cout << "Beta = " << beta << endl;
        
        // Update search direction: p = r + beta * p
        for (int i = 0; i < M; ++i) {
            p[i] = r[i] + beta * p[i];
        }
        
        // Update rsold for next iteration
        rsold = rsnew;
        
        // Safety check for maximum iterations
        if (k >= 2 * M) {
            cout << "Maximum iterations (" << 2*M << ") reached, stopping" << endl;
            break;
        }
        
        // Debug: Check solution progress every 10 iterations
        if (k % 10 == 0 && k > 0) {
            long double weightSum = 0.0L;
            for(int i = 0; i < N; i++) {
                weightSum += x[i];
            }
            cout << "Progress check - Current weight sum: " << weightSum << endl;
        }
    }
    
    cout << "Finish processing Conjugate Gradient Program......" << endl;
    
    // Step 5: Final verification
    Vec final_Qx = MatM_Vec(Q, x);
    long double final_residual_norm = 0.0L;
    for (int i = 0; i < M; i++) {
        long double diff = b[i] - final_Qx[i];
        final_residual_norm += diff * diff;
    }
    final_residual_norm = sqrt(final_residual_norm);
    
    cout << "Final residual norm: " << final_residual_norm << endl;
    
    // Check if any weights are NaN or Inf
    bool hasInvalidValues = false;
    for(int i = 0; i < M; i++) {
        if(isnan(x[i]) || isinf(x[i])) {
            cout << "Warning: Invalid value detected at x[" << i << "] = " << x[i] << endl;
            hasInvalidValues = true;
        }
    }
    
    if(hasInvalidValues) {
        cout << "ERROR: Solution contains NaN or Inf values!" << endl;
        // Return a reasonable fallback solution
        Vec fallback(M, 0.0L);
        for(int i = 0; i < N; i++) {
            fallback[i] = 1.0L / N;  // Equal weights
        }
        fallback[N] = 0.01L;
        fallback[N+1] = 1.0L;
        cout << "Returning fallback equal-weight solution" << endl;
        return fallback;
    }
    
    // Final solution summary
    long double weightSum = 0.0L;
    int negativeWeights = 0;
    for(int i = 0; i < N; i++) {
        weightSum += x[i];
        if(x[i] < 0) negativeWeights++;
    }
    
    cout << "\n\n\n=== SOLUTION SUMMARY ===" << endl;
    cout << "Portfolio weight sum: " << weightSum << " (should be ~1.0)" << endl;
    cout << "Number of negative weights (short positions): " << negativeWeights << endl;
    cout << "Lambda (return multiplier): " << x[N] << endl;
    cout << "Mu (budget multiplier): " << x[N+1] << endl;
    cout << "Final residual norm: " << final_residual_norm << "\n\n" <<endl;
    
    return x;
}



// Function to write Matrix2D to CSV file
void PortfolioOptimizer::writeMatrixToCSV(const Matrix2D& matrix, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "Error: Could not open file " << filename << " for writing." << endl;
        return;
    }
    
    // Set precision for floating point numbers
    file << fixed << setprecision(10);
    
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = 0; j < matrix[i].size(); ++j) {
            file << matrix[i][j];
            if (j < matrix[i].size() - 1) {
                file << ",";  // Add comma between values
            }
        }
        file << "\n";  // New line after each row
    }
    
    file.close();
    cout << "Matrix written to " << filename << endl;
}

// Function to write Vector to CSV file
void PortfolioOptimizer::writeVectorToCSV(const Vec& vector, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "Error: Could not open file " << filename << " for writing." << endl;
        return;
    }
    
    file << fixed << setprecision(10);
    
    for (int i = 0; i < vector.size(); ++i) {
        file << vector[i] << "\n";
    }
    
    file.close();
    cout << "Vector written to " << filename << endl;
}



