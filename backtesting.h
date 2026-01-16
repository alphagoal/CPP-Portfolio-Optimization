#ifndef backtesting_h
#define backtesting_h

#include <vector>
#include <string>

using namespace std;
using Vec = vector<long double>;
using Matrix2D = vector<vector<long double>>;
using Cube = vector<Matrix2D>;

struct PortfolioResult {
    long double targetReturn;
    long double actualReturn;           // In-sample theoretical return
    long double averageOutOfSampleReturn;
    long double portfolioVariance;      // In-sample theoretical variance
    long double outOfSampleVariance;    // Empirical out-of-sample variance
    Vec weights;
    vector<long double> outOfSampleReturns;
    int numWindows;
};

class Backtester {
private:
    // Core data - references to pre-computed results
    const Matrix2D& returns_;
    const Matrix2D& meanReturns_;
    const Cube& covarianceReturns_;
    
    // Parameters
    int windowSize_;
    int outOfSamplePeriod_;
    
    // Derived parameters
    int numAssets_;
    int totalPeriods_;
    int totalWindows_;
    
    // Results
    vector<PortfolioResult> results_;
    
    // Helper functions
    PortfolioResult optimizePortfolio(long double targetReturn, int endPeriod);
    vector<long double> calculateOutOfSampleReturns(const Vec& weights, int startPeriod);
    long double calculatePortfolioReturn(const Vec& weights, const Vec& assetReturns) const;
    long double calculatePortfolioVariance(const Vec& weights, const Matrix2D& covarianceMatrix) const;
    vector<long double> generateTargetReturns(long double minReturn, long double maxReturn) const;
    void calculatePortfolioStatistics();

public:
    // Constructor - takes pre-computed data
    Backtester(const Matrix2D& returns, 
               const Matrix2D& meanReturns,
               const Cube& covarianceReturns,
               int windowSize, 
               int outOfSamplePeriod);
    
    // Main backtesting function
    void runBacktest(long double minReturn, long double maxReturn);
    
    // Output functions
    void printSummary() const;
    void printDetailedAnalysis() const;
    void exportResults(const string& filename) const;
};

#endif