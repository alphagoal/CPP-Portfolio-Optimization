#include "csv.h"
#include "port_optimization.h"
#include "para_estimation.h"
#include "backtester.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <numeric>  

using namespace std;
using Vec = vector <long double>;
using Matrix2D = vector<vector<long double> >;
using Cube     = vector<Matrix2D>;  

// ******* CONSTRUCTER *******
Backtester::Backtester(int totalPeriods, int windowSize, int outOfSampleSize, int numberAssets)
    : totalPeriods_(totalPeriods), windowSize_(windowSize), 
      outOfSampleSize_(outOfSampleSize), numberAssets_(numberAssets) {
    
    cout << "Backtester initialized with:" << endl;
    cout << "- Total periods: " << totalPeriods_ << endl;
    cout << "- Window size: " << windowSize_ << endl;
    cout << "- Out-of-sample size: " << outOfSampleSize_ << endl;
    cout << "- Number of assets: " << numberAssets_ << endl;
}

// ******* FUNCTION 1 *******
//Load data and compute rolling means and covariance
void Backtester::computeRollingParameters() {
    
    // Allocate memory for return data
    long double **returnMatrix = new long double*[numberAssets_];
    for(int i = 0; i < numberAssets_; i++) {
        returnMatrix[i] = new long double[totalPeriods_];
    }
    
    // Read the data from file
    string fileName = "asset_returns.csv";
    ParameterEstimator::readData(returnMatrix, fileName);
    
    // Transform to vector object
    returns_vec_ = ParameterEstimator::transform_to_vec(returnMatrix, numberAssets_, totalPeriods_);
    
    // Compute rolling means
    cout << "\nProcessing rolling means ....";
    mean_returns_ = ParameterEstimator::rollingMeans(returns_vec_);
    
    // Compute rolling covariance
    cout << "\nProcessing rolling covariance ....";
    covariance_returns_ = ParameterEstimator::rollingCovariance(returns_vec_, mean_returns_);
    
    // Clean up memory
    for(int i = 0; i < numberAssets_; i++) {
        delete[] returnMatrix[i];
    }
    delete[] returnMatrix;

    cout << "\nRolling parameters computed successfully!" << endl;
}

// ******* FUNCTION 2 *******
//Get portfolio weights for one specific period and target return
Vec Backtester::getPortfolioWeights(int period, long double targetReturn) {
    
    cout << "Processing weights for period " << period << " (target " << targetReturn << ")..." << endl;
    
    // Extract covariance matrix for this period
    Matrix2D Sigma = covariance_returns_[period];
    
    // Extract mean returns for this period
    Vec meanReturn(numberAssets_);
    for(int i = 0; i < numberAssets_; ++i) {
        meanReturn[i] = mean_returns_[i][period];
    }
    
    // Build KKT matrix and RHS vector
    Matrix2D Q = PortfolioOptimizer::KKTMatrix_Q(Sigma, meanReturn);
    Vec b = PortfolioOptimizer::RHS_b(numberAssets_, targetReturn);
    
    // Solve for weights
    Vec x = PortfolioOptimizer::solveKKT_x(Q, b);
    
    // Extract just the portfolio weights (first n elements)
    Vec weights(numberAssets_);
    for(int i = 0; i < numberAssets_; ++i) {
        weights[i] = x[i];
    }
    

    cout << "\n\n\nWeights for period " << period << " (target " << targetReturn << "):\n";
    for (int i =0; i <numberAssets_;++i) {
        cout<< "Asset "<< i <<" weight:" << x[i]<<endl;
    }

    // Verify weights sum to 1
    long double sum_w = 0.0L;
    for(int i = 0; i < numberAssets_; ++i) {
        sum_w += weights[i];
    }
    cout << "Sum of weights = " << sum_w << endl;
    
    return weights;
}


// ******* FUNCTION 3 *******
// Print data check (utility function)
void Backtester::printDataCheck() {
    cout << "Checking:\n\n\nChecking rolling means ....";
    cout << "\nCheck mean returns - Asset 0, Period 98 : " << mean_returns_[0][98];
    cout << "\nCheck mean returns - Asset 0, Period 99 : " << mean_returns_[0][99];
    cout << "\nCheck mean returns - Asset 82, Period 699 : " << mean_returns_[82][699];
    cout << "\nCheck mean returns - Asset 82, Period 500 : " << mean_returns_[82][500] << endl;
    cout << "\nChecking rolling covariance ....";
    cout << "\nCheck covariance - Period 98, Comp0, Comp2 : " << covariance_returns_[98][0][2];
    cout << "\nCheck covariance - Period 99, Comp0, Comp2: " << covariance_returns_[99][0][2];
    cout << "\nCheck covariance - Period 99, Comp0, Comp0: " << covariance_returns_[99][0][0];
    cout << "\nCheck covariance - Period 99, Comp2, Comp0: " << covariance_returns_[99][2][0];
    cout << "\nCheck covariance - Period 99, Comp17, Comp0: " << covariance_returns_[99][17][0];
    cout << "\nCheck covariance - Period 99, Comp41, Comp82: " << covariance_returns_[99][41][82];
    cout << "\nCheck covariance - Period 99, Comp82, Comp41: " << covariance_returns_[99][82][41];
    cout << "\nCheck covariance - Period 99, Comp82, Comp82: " << covariance_returns_[99][82][82] << endl;
}


// ******* FUNCTION 4 *******
//Identify which day point I need to solve the weights. 
//For example, if I want 99th period (100th day), then add 12days until 699-12= 687th period (688th day)
vector<int> Backtester::getEvaluationDates() {
    vector<int> endPeriods;
    int firstEnd = windowSize_ - 1;  // 99
    int lastEnd = totalPeriods_ - outOfSampleSize_ - 1;  // 687
    
    for (int end = firstEnd; end <= lastEnd; end += outOfSampleSize_) {
        endPeriods.push_back(end);
    }

    //Print Evaluating Start Dates for Each Window:
    cout << "\nEvaluation dates: ";
    for (int i = 0; i < endPeriods.size(); i++) {
        cout << endPeriods[i];
        if (i < endPeriods.size() - 1) cout << ", ";
    }
    cout << "\nTotal windows: " << endPeriods.size() << endl;

    return endPeriods;
}

/* ===============================================================================
                                Column indices:
[0] = Target return, [1] = Daily return, [2] = Daily variance, [3] = Daily std dev
[4] = Annual return, [5] = Annual variance, [6] = Annual std dev
[7] = Sharpe ratio, [8] = Number of periods
=================================================================================== */

// ******* FUNCTION 5 *******
// Expanded function that creates both window-based and daily return matrices
ComprehensiveBacktestingResults Backtester::createComprehensiveBacktesting(long double riskFreeRate) {
    vector<int> evaluationDates = getEvaluationDates();
    
    // Generate target returns: 0.0%, 0.5%, 1.0%, ..., 10.0%
    vector<long double> targetReturns;
    for (int i = 0; i <= 20; i++) {
        targetReturns.push_back(i * 0.005);
    }
    
    cout << "Creating comprehensive backtesting results..." << endl;
    cout << "Target returns: " << targetReturns.size() << endl;
    cout << "Evaluation dates: " << evaluationDates.size() << endl;
    cout << "Days per window: " << outOfSampleSize_ << endl;
    
    // Calculate total days (50 windows × 12 days = 600 days)
    int totalDays = evaluationDates.size() * outOfSampleSize_;
    cout << "Total out-of-sample days: " << totalDays << endl;
    
    //CONSTRUCT DATA TYPE results (include all 4 datatypes)
    ComprehensiveBacktestingResults results;
    
    // INITIALIze matrices according to "struct" (smart way to declare with struct)
    results.dailyReturns = Matrix2D(targetReturns.size(), Vec(totalDays, 0.0L));
    results.windowMeanReturns = Matrix2D(targetReturns.size(), Vec(evaluationDates.size(), 0.0L));
    results.windowVariances = Matrix2D(targetReturns.size(), Vec(evaluationDates.size(), 0.0L));
    results.portfolioMetrics.resize(targetReturns.size(), Vec(9, 0.0L));  // 21×9 matrix
    
    // Loop through each target return - t means target yearly return
    for (int t = 0; t < targetReturns.size(); t++) {
        long double targetReturn = targetReturns[t];
        
        cout << "\nProcessing target return " << (t+1) << "/" << targetReturns.size() 
             << ": " << targetReturn*100 << "%" << endl;
        
        vector<long double> allDailyReturns;  // Collect all 600 daily returns for this target
        allDailyReturns.reserve(totalDays); //reserve memory space for push_back, so here reserve 600 days 
        
        int dayIndex = 0;  // Index for daily returns matrix
        
        // WINDOW BY WINDOW
        // Loop through each evaluation period (50 windows)
        for (int p = 0; p < evaluationDates.size(); p++) {
            int period = evaluationDates[p];
            
            if (p < 5 || p % 10 == 0) {
                cout << "  Window " << (p+1) << "/" << evaluationDates.size() 
                     << " (rebalance at day " << period+1 << ")" << endl;
            }
            
            // Get NEW portfolio weights for this window (rebalancing every 12 days!)
            Vec weights = Backtester::getPortfolioWeights(period, targetReturn);
            
            // Calculate out-of-sample returns using these weights for next 12 days
            Vec windowReturns; //create a vector of 12 days (in next line) --> re-declare in each window
            windowReturns.reserve(outOfSampleSize_);
            
            for (int day = 1; day <= outOfSampleSize_; day++) {
                int returnPeriod = period + day;
                
                // Calculate portfolio return for this day
                long double dailyReturn = 0.0L;
                for (int asset = 0; asset < numberAssets_; asset++) {
                    dailyReturn += weights[asset] * returns_vec_[asset][returnPeriod];
                }
                
                // Store in both matrices
                results.dailyReturns[t][dayIndex] = dailyReturn;  // Daily returns matrix
                windowReturns.push_back(dailyReturn); // this is declared within the for loop
                allDailyReturns.push_back(dailyReturn); //this is declared earlier already
                
                dayIndex++;
            }
            
            // ********** Calculate WINDOW STATISTICS (mean and variance for this 12-day period) **********
            long double windowSum = 0.0L;
            for (long double ret : windowReturns) {
                windowSum += ret;
            }
            long double windowMean = windowSum / windowReturns.size();
            
            long double windowVariance = 0.0L;
            for (long double ret : windowReturns) {
                long double diff = ret - windowMean;
                windowVariance += diff * diff;
            }
            if (windowReturns.size() > 1) {
                windowVariance /= (windowReturns.size() - 1);
            }
            
            // Store window statistics
            results.windowMeanReturns[t][p] = windowMean;
            results.windowVariances[t][p] = windowVariance;
        }
        // ********** ABOVE already calculated WINDOW STATISTICS (mean and variance for this 12-day period) **********


        // ********** Start calculating ALL-TIME STATISTICS (mean and variance for this 12-day period) **********
        // Calculate comprehensive portfolio metrics using ALL 600 daily returns
        int numPeriods = allDailyReturns.size(); //600 DAYS
        
        // Calculate overall mean return
        long double totalSum = 0.0L;
        for (long double ret : allDailyReturns) {
            totalSum += ret;
        }
        long double dailyMeanReturn = totalSum / allDailyReturns.size();
        
        // Calculate overall variance (true 600-day variance with rebalancing)
        long double variance = 0.0L;
        for (long double ret : allDailyReturns) {
            long double diff = ret - dailyMeanReturn;
            variance += diff * diff;
        }
        if (allDailyReturns.size() > 1) {
            variance /= (allDailyReturns.size() - 1);
        }
        long double totalStdDev = sqrt(variance);
        
        // Annualize metrics (252 trading days per year)
        long double annualizedReturn = dailyMeanReturn * 252.0L;
        long double annualizedVariance = variance * 252.0L;
        long double annualizedStdDev = sqrt(annualizedVariance);
        
        // Calculate Sharpe ratio
        long double sharpeRatio = 0.0L;
        if (annualizedStdDev > 1e-8) {
            sharpeRatio = (annualizedReturn - riskFreeRate) / annualizedStdDev;
        }
        
        // Store comprehensive portfolio metrics in Matrix2D format
        results.portfolioMetrics[t][0] = targetReturn;           // Target return
        results.portfolioMetrics[t][1] = dailyMeanReturn;            // Daily mean return
        results.portfolioMetrics[t][2] = variance;               // Daily variance
        results.portfolioMetrics[t][3] = totalStdDev;            // Daily std dev
        results.portfolioMetrics[t][4] = annualizedReturn;       // Annual return
        results.portfolioMetrics[t][5] = annualizedVariance;     // Annual variance
        results.portfolioMetrics[t][6] = annualizedStdDev;       // Annual std dev
        results.portfolioMetrics[t][7] = sharpeRatio;            // Sharpe ratio
        results.portfolioMetrics[t][8] = (long double)numPeriods; // Number of periods
        
        cout << "  → Daily return: " << fixed << setprecision(4) << dailyMeanReturn*100 
             << "%, Annual return: " << annualizedReturn*100 
             << "%, Sharpe: " << sharpeRatio << endl;
    }
    
    cout << "\n=== COMPREHENSIVE BACKTESTING COMPLETED ===" << endl;
    cout << "Daily returns matrix: " << results.dailyReturns.size() << " × " << results.dailyReturns[0].size() << endl;
    cout << "Window statistics: " << results.windowMeanReturns.size() << " × " << results.windowMeanReturns[0].size() << endl;
    
    //  =========== Export all CSV files =========== 
    cout << "\nExporting results to CSV files..." << endl;
    
    // 1. 21 × 50 window return statistics
    PortfolioOptimizer::writeMatrixToCSV(results.windowMeanReturns, "21x50_window_returns.csv");
    cout << "Exported: 21x50_window_returns.csv" << endl;
    
    // 2. 21 × 50 window variance statistics  
    PortfolioOptimizer::writeMatrixToCSV(results.windowVariances, "21x50_window_variances.csv");
    cout << "Exported: 21x50_window_variances.csv" << endl;
    
    // 3. 21 × 600 daily return statistics
    PortfolioOptimizer::writeMatrixToCSV(results.dailyReturns, "21x600_daily_returns.csv");
    cout << "Exported: 21x600_daily_returns.csv" << endl;
    
    // 4. 21 × N portfolio metrics (already in matrix format)
    PortfolioOptimizer::writeMatrixToCSV(results.portfolioMetrics, "21xN_portfolio_metrics.csv");
    cout << "Exported: 21xN_portfolio_metrics.csv" << endl;
    
    // Create header file for portfolio metrics CSV
    ofstream headerFile("portfolio_metrics_header.txt");
    if (headerFile.is_open()) {
        headerFile << "Column headers for 21xN_portfolio_metrics.csv:" << endl;
        headerFile << "Column 0: Target Return (decimal)" << endl;
        headerFile << "Column 1: Daily Mean Return (decimal)" << endl;
        headerFile << "Column 2: Daily Variance" << endl;
        headerFile << "Column 3: Daily Standard Deviation" << endl;
        headerFile << "Column 4: Annualized Return (decimal)" << endl;
        headerFile << "Column 5: Annualized Variance" << endl;
        headerFile << "Column 6: Annualized Standard Deviation" << endl;
        headerFile << "Column 7: Sharpe Ratio" << endl;
        headerFile << "Column 8: Number of Periods" << endl;
        headerFile.close();
        cout << "✓ Exported: portfolio_metrics_header.txt" << endl;
    }
    
    cout << "\nAll CSV files exported successfully!" << endl;
    cout << "File Summary for User:" << endl;
    cout << "- 21x50_window_returns.csv: Mean returns for each window" << endl;
    cout << "- 21x50_window_variances.csv: Variances for each window" << endl;
    cout << "- 21x600_daily_returns.csv: Daily portfolio returns with rebalancing" << endl;
    cout << "- 21xN_portfolio_metrics.csv: Comprehensive portfolio statistics" << endl;
    cout << "- portfolio_metrics_header.txt: Column descriptions for metrics file" << endl;
    
    return results;
}

// ******* FUNCTION 6 *******
// Function to print comprehensive summary for reference
void Backtester::printComprehensiveSummary(const ComprehensiveBacktestingResults& results, 
                              long double riskFreeRate) {
    
    cout << "\n" << string(100, '=') << endl;
    cout << "COMPREHENSIVE PORTFOLIO PERFORMANCE SUMMARY" << endl;
    cout << string(100, '=') << endl;
    
    cout << "Risk-free rate: " << riskFreeRate*100 << "%" << endl;
    cout << "Rebalancing: Every 12 days (50 rebalancing periods)" << endl;
    cout << "Analysis period: 600 out-of-sample days" << endl;
    cout << endl;
    
    cout << setw(8) << "Target%" << setw(12) << "Daily%" << setw(12) << "Annual%" 
         << setw(12) << "Ann.StdDev%" << setw(12) << "Sharpe" << setw(12) << "Days" << endl;
    cout << string(100, '-') << endl;
    
    for (int t = 0; t < results.portfolioMetrics.size(); t++) {
        long double targetReturn = t * 0.005;

        cout << fixed << setprecision(1);
        cout << setw(8) << targetReturn * 100;
        cout << setprecision(3);
        cout << setw(12) << results.portfolioMetrics[t][1] * 100;  // Daily return
        cout << setprecision(2);
        cout << setw(12) << results.portfolioMetrics[t][4] * 100;  // Annual return
        cout << setw(12) << results.portfolioMetrics[t][6] * 100;  // Annual std dev
        cout << setprecision(3);
        cout << setw(12) << results.portfolioMetrics[t][7];        // Sharpe ratio
        cout << setw(12) << (int)results.portfolioMetrics[t][8];   // Num periods
        cout << endl;
    }
    
    cout << string(100, '-') << endl;
    
    // Find best performing portfolios
    int bestSharpeIndex = 0, bestReturnIndex = 0;
    long double maxSharpe = results.portfolioMetrics[0][7];
    long double maxReturn = results.portfolioMetrics[0][4];
    
    for (int t = 1; t < results.portfolioMetrics.size(); t++) {
        if (results.portfolioMetrics[t][7] > maxSharpe) {
            maxSharpe = results.portfolioMetrics[t][7];
            bestSharpeIndex = t;
        }
        if (results.portfolioMetrics[t][4] > maxReturn) {
            maxReturn = results.portfolioMetrics[t][4];
            bestReturnIndex = t;
        }
    }
    
    cout << "\nKEY FINDINGS:" << endl;
    cout << "Best Sharpe Ratio: " << setprecision(3) << maxSharpe 
         << " (Target: " << bestSharpeIndex*0.5 << "%)" << endl;
    cout << "Highest Return: " << setprecision(2) << maxReturn*100 
         << "% (Target: " << bestReturnIndex*0.5 << "%)" << endl;
    
    cout << string(100, '=') << endl;
}

