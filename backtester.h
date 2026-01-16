#ifndef backtester_h
#define backtester_h

#include <iostream>
#include <cmath>
#include <vector>

using namespace std;
using Vec = vector <long double>;
using Matrix2D = vector<vector<long double> >;
using Cube     = vector<Matrix2D>;  

struct ComprehensiveBacktestingResults {
    Matrix2D dailyReturns;        // [target_return][day] - 21×600 matrix of daily returns
    Matrix2D windowMeanReturns;   // [target_return][period] - original 21×50 matrix
    Matrix2D windowVariances;     // [target_return][period] - original 21×50 matrix
    Matrix2D portfolioMetrics;    // [target_return][metric] - 21×9 matrix of portfolio metrics
};

class Backtester {
private:
    // Data storage
    int totalPeriods_;
    int windowSize_;
    int outOfSampleSize_;
    int numberAssets_;
    Matrix2D returns_vec_;
    Matrix2D mean_returns_;
    Cube covariance_returns_;
    
public:
    // Constructor
    Backtester(int totalPeriods, int windowSize, int outOfSampleSize, int numberAssets);
    
    // Main functions (public because they're used from main)
    void computeRollingParameters();
    ComprehensiveBacktestingResults createComprehensiveBacktesting(long double riskFreeRate = 0.02L);
    void printComprehensiveSummary(const ComprehensiveBacktestingResults& results, long double riskFreeRate = 0.02L);
    
    //get functions
    vector<int> getEvaluationDates();
    Vec getPortfolioWeights(int period, long double targetReturn);

    // Utility functions
    void printDataCheck();


};

#endif