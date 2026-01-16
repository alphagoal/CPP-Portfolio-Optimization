#include "backtesting.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

using namespace std;

// Constructor
Backtester::Backtester(const Matrix2D& returns, int windowSize, int outOfSamplePeriod) 
    : returns_(returns), windowSize_(windowSize), outOfSamplePeriod_(outOfSamplePeriod) { //ensure these variables exist in the entire object lifetime
    
    numAssets_ = returns_.size();
    totalPeriods_ = returns_[0].size();
    int firstStartPeriod = windowSize_ - 1;  // 99
    int lastStartPeriod = totalPeriods_ - outOfSamplePeriod_ - 1;  // 700 - 12 - 1 = 687
    totalWindows_ = (lastStartPeriod - firstStartPeriod) / outOfSamplePeriod_ + 1;
    
    
    cout << "\nBacktester initialized:" << endl;
    cout << "- Assets: " << numAssets_ << endl;
    cout << "- Total periods: " << totalPeriods_ << endl;
    cout << "- Window size: " << windowSize_ << endl;
    cout << "- Out-of-sample period: " << outOfSamplePeriod_ << endl;
    cout << "- Total rolling windows: " << totalWindows_ << endl;
}

// Main backtesting function
void Backtester::runBacktest(int numTargetReturns, long double minReturn, long double maxReturn) {
    cout << "\n=== STARTING BACKTESTING PROCEDURE ===" << endl;
    cout << "Target returns: " << numTargetReturns << " portfolios from " 
         << minReturn*100 << "% to " << maxReturn*100 << "%" << endl;
    
    // Generate target returns
    vector<long double> targetReturns = generateTargetReturns(numTargetReturns, minReturn, maxReturn);
    
    // Clear previous results
    results_.clear();
    results_.reserve(numTargetReturns);
    
    // For each target return, run the full backtesting procedure
    for (int i = 0; i < numTargetReturns; i++) {
        long double targetReturn = targetReturns[i];
        cout << "\n--- Processing portfolio " << (i+1) << "/" << numTargetReturns 
             << " (target return: " << targetReturn*100 << "%) ---" << endl;
        
        // Initialize portfolio result
        PortfolioResult result;
        result.targetReturn = targetReturn;
        result.outOfSampleReturns.reserve(totalWindows_);
        
        // Roll through all windows
        int windowCount = 0;
        for (int startPeriod = windowSize_ - 1; 
             startPeriod <= totalPeriods_ - windowSize_ - outOfSamplePeriod_; 
             startPeriod += outOfSamplePeriod_) {
            
            windowCount++;
            if (windowCount % 10 == 1 || windowCount <= 5) {
                cout << "  Window " << windowCount << ": periods " << startPeriod-windowSize_+1 
                     << " to " << startPeriod << " (in-sample)" << endl;
            }
            
            // Optimize portfolio for this window
            PortfolioResult windowResult = optimizePortfolio(targetReturn, startPeriod);
            
            // Store weights from first window (or update with latest)
            result.weights = windowResult.weights;
            
            // Calculate out-of-sample returns for next 12 periods
            vector<long double> oosReturns = calculateOutOfSampleReturns(windowResult.weights, startPeriod + 1);
            
            // Store individual out-of-sample returns
            result.outOfSampleReturns.insert(result.outOfSampleReturns.end(), 
                                           oosReturns.begin(), oosReturns.end());
        }
        
        result.numWindows = windowCount;
        
        // Calculate average out-of-sample return
        if (!result.outOfSampleReturns.empty()) {
            long double sum = 0.0L;
            for (long double ret : result.outOfSampleReturns) {
                sum += ret;
            }
            result.averageOutOfSampleReturn = sum / result.outOfSampleReturns.size();
        }
        
        cout << "  Completed " << windowCount << " windows, average OOS return: " 
             << result.averageOutOfSampleReturn * 100 << "%" << endl;
        
        results_.push_back(result);
    }
    
    // Calculate portfolio statistics (variance, etc.)
    calculatePortfolioStatistics();
    
    cout << "\n=== BACKTESTING COMPLETED ===" << endl;
}

// Generate evenly spaced target returns
vector<long double> Backtester::generateTargetReturns(int numTargets, long double minReturn, long double maxReturn) const {
    vector<long double> targets;
    // Calculate increment to achieve 0.5% spacing
    double increment = 0.005; // 0.5% increment
    
    // Generate targets from minReturn to maxReturn in 0.5% steps
    for (double target = minReturn; target <= maxReturn; target += increment) {
        targets.push_back(target);
    }
    
    return targets;
}

// Optimize portfolio for a specific period and target return
PortfolioResult Backtester::optimizePortfolio(long double targetReturn, int endPeriod) {
    PortfolioResult result;
    result.targetReturn = targetReturn;
    
    // Extract in-sample window
    int startPeriod = endPeriod - windowSize_ + 1;
    Matrix2D windowReturns = extractWindow(startPeriod, endPeriod);
    
    // Calculate rolling means and covariance for this window
    Matrix2D meanReturns = ParameterEstimator::rollingMeans(windowReturns, windowSize_);
    Cube covarianceReturns = ParameterEstimator::rollingCovariance(windowReturns, meanReturns, windowSize_);
    
    // Get the covariance matrix and mean returns for the last period of the window
    Matrix2D Sigma = covarianceReturns[windowSize_ - 1];
    Vec meanReturn(numAssets_);
    for (int i = 0; i < numAssets_; i++) {
        meanReturn[i] = meanReturns[i][windowSize_ - 1];
    }
    
    // Solve optimization problem
    Matrix2D Q = PortfolioOptimizer::KKTMatrix_Q(Sigma, meanReturn);
    Vec b = PortfolioOptimizer::RHS_b(numAssets_, targetReturn);
    Vec x = PortfolioOptimizer::solveKKT_x(Q, b);
    
    // Extract portfolio weights (first numAssets_ elements)
    result.weights.resize(numAssets_);
    for (int i = 0; i < numAssets_; i++) {
        result.weights[i] = x[i];
    }
    
    // Calculate in-sample portfolio return
    result.actualReturn = calculatePortfolioReturn(result.weights, meanReturn);
    
    // Calculate portfolio variance
    result.portfolioVariance = calculatePortfolioVariance(result.weights, Sigma);
    
    return result;
}







// Calculate out-of-sample returns for the next outOfSamplePeriod_ periods
vector<long double> Backtester::calculateOutOfSampleReturns(const Vec& weights, int startPeriod) {
    vector<long double> oosReturns;
    oosReturns.reserve(outOfSamplePeriod_);
    
    for (int t = startPeriod; t < startPeriod + outOfSamplePeriod_ && t < totalPeriods_; t++) {
        // Get asset returns for period t
        Vec assetReturns(numAssets_);
        for (int i = 0; i < numAssets_; i++) {
            assetReturns[i] = returns_[i][t];
        }
        
        // Calculate portfolio return
        long double portfolioReturn = calculatePortfolioReturn(weights, assetReturns);
        oosReturns.push_back(portfolioReturn);
    }
    
    return oosReturns;
}

// Calculate portfolio return: w^T * r
long double Backtester::calculatePortfolioReturn(const Vec& weights, const Vec& assetReturns) const {
    long double portfolioReturn = 0.0L;
    for (int i = 0; i < numAssets_; i++) {
        portfolioReturn += weights[i] * assetReturns[i];
    }
    return portfolioReturn;
}

// Calculate portfolio variance: w^T * Σ * w
long double Backtester::calculatePortfolioVariance(const Vec& weights, const Matrix2D& covarianceMatrix) const {
    long double variance = 0.0L;
    for (int i = 0; i < numAssets_; i++) {
        for (int j = 0; j < numAssets_; j++) {
            variance += weights[i] * weights[j] * covarianceMatrix[i][j];
        }
    }
    return variance;
}

// Calculate portfolio statistics (variance using out-of-sample data)
void Backtester::calculatePortfolioStatistics() {
    cout << "\nCalculating portfolio statistics..." << endl;
    
    for (auto& result : results_) {
        if (!result.outOfSampleReturns.empty()) {
            // Calculate variance of out-of-sample returns
            long double mean = result.averageOutOfSampleReturn;
            long double variance = 0.0L;
            
            for (long double ret : result.outOfSampleReturns) {
                long double diff = ret - mean;
                variance += diff * diff;
            }
            
            if (result.outOfSampleReturns.size() > 1) {
                variance /= (result.outOfSampleReturns.size() - 1);
            }
            
            result.portfolioVariance = variance;
        }
    }
}

// Extract a window of returns
Matrix2D Backtester::extractWindow(int startPeriod, int endPeriod) const {
    validatePeriods(startPeriod, endPeriod);
    
    int windowLength = endPeriod - startPeriod + 1;
    Matrix2D window(numAssets_, vector<long double>(windowLength));
    
    for (int i = 0; i < numAssets_; i++) {
        for (int t = 0; t < windowLength; t++) {
            window[i][t] = returns_[i][startPeriod + t];
        }
    }
    
    return window;
}

// Validate period indices
void Backtester::validatePeriods(int startPeriod, int endPeriod) const {
    if (startPeriod < 0 || endPeriod >= totalPeriods_ || startPeriod > endPeriod) {
        throw invalid_argument("Invalid period range: [" + to_string(startPeriod) + 
                             ", " + to_string(endPeriod) + "]");
    }
}

// Print comprehensive summary
void Backtester::printSummary() const {
    cout << "\n" << string(80, '=') << endl;
    cout << "                    BACKTESTING RESULTS SUMMARY" << endl;
    cout << string(80, '=') << endl;
    
    cout << "Dataset Information:" << endl;
    cout << "- Number of assets: " << numAssets_ << endl;
    cout << "- Total periods: " << totalPeriods_ << endl;
    cout << "- Rolling windows: " << totalWindows_ << endl;
    cout << "- Window size: " << windowSize_ << " days" << endl;
    cout << "- Out-of-sample period: " << outOfSamplePeriod_ << " days" << endl;
    cout << endl;
    
    cout << "Portfolio Performance Analysis:" << endl;
    cout << string(85, '-') << endl;
    cout << setw(10) << "Target" << setw(12) << "Actual" << setw(12) << "Difference" 
         << setw(14) << "Variance" << setw(12) << "Std Dev" << setw(12) << "Sharpe" << endl;
    cout << setw(10) << "Return(%)" << setw(12) << "Return(%)" << setw(12) << "(%)" 
         << setw(14) << "" << setw(12) << "(%)" << setw(12) << "Ratio" << endl;
    cout << string(85, '-') << endl;
    
    for (const auto& result : results_) {
        long double difference = result.averageOutOfSampleReturn - result.targetReturn;
        long double stdDev = sqrt(result.portfolioVariance);
        long double sharpe = (stdDev > 1e-8) ? result.averageOutOfSampleReturn / stdDev : 0.0L;
        
        cout << fixed << setprecision(2);
        cout << setw(10) << result.targetReturn * 100;
        cout << setw(12) << result.averageOutOfSampleReturn * 100;
        cout << setw(12) << difference * 100;
        cout << setw(14) << setprecision(6) << result.portfolioVariance;
        cout << setw(12) << setprecision(2) << stdDev * 100;
        cout << setw(12) << setprecision(3) << sharpe << endl;
    }
    
    cout << string(85, '-') << endl;
}

// Print compact table
void Backtester::printCompactTable() const {
    cout << "\n=== PORTFOLIO PERFORMANCE TABLE ===" << endl;
    cout << setw(8) << "Target%" << setw(10) << "Actual%" << setw(12) << "Variance" 
         << setw(10) << "Std Dev%" << endl;
    cout << string(40, '-') << endl;
    
    for (const auto& result : results_) {
        cout << fixed << setprecision(2);
        cout << setw(8) << result.targetReturn * 100;
        cout << setw(10) << result.averageOutOfSampleReturn * 100;
        cout << setw(12) << setprecision(6) << result.portfolioVariance;
        cout << setw(10) << setprecision(2) << sqrt(result.portfolioVariance) * 100 << endl;
    }
    cout << string(40, '-') << endl;
}

// Export results to CSV
void Backtester::exportResults(const string& filename) const {
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "Error: Could not open " << filename << " for writing." << endl;
        return;
    }
    
    file << "TargetReturn,ActualReturn,Variance,StdDev,NumWindows" << endl;
    file << fixed << setprecision(8);
    
    for (const auto& result : results_) {
        file << result.targetReturn << ","
             << result.averageOutOfSampleReturn << ","
             << result.portfolioVariance << ","
             << sqrt(result.portfolioVariance) << ","
             << result.numWindows << endl;
    }
    
    file.close();
    cout << "Results exported to " << filename << endl;
}

// Export for plotting (efficient frontier and target vs actual)
void Backtester::exportForPlotting(const string& efficiencyFrontierFile, const string& targetVsActualFile) const {
    // Efficient frontier: Mean return vs Standard deviation
    ofstream efFile(efficiencyFrontierFile);
    if (efFile.is_open()) {
        efFile << "StdDev,MeanReturn" << endl;
        efFile << fixed << setprecision(8);
        
        for (const auto& result : results_) {
            efFile << sqrt(result.portfolioVariance) << ","
                   << result.averageOutOfSampleReturn << endl;
        }
        efFile.close();
        cout << "Efficient frontier data exported to " << efficiencyFrontierFile << endl;
    }
    
    // Target vs Actual returns
    ofstream taFile(targetVsActualFile);
    if (taFile.is_open()) {
        taFile << "TargetReturn,ActualReturn" << endl;
        taFile << fixed << setprecision(8);
        
        for (const auto& result : results_) {
            taFile << result.targetReturn << ","
                   << result.averageOutOfSampleReturn << endl;
        }
        taFile.close();
        cout << "Target vs Actual data exported to " << targetVsActualFile << endl;
    }
}

// Simple terminal plot
void Backtester::plotInTerminal() const {
    cout << "\n" << string(60, '=') << endl;
    cout << "                EFFICIENT FRONTIER PLOT" << endl;
    cout << string(60, '=') << endl;
    cout << "Risk (Standard Deviation %) vs Return (%)" << endl;
    cout << string(60, '-') << endl;
    
    // Find min/max for scaling
    long double minReturn = results_[0].averageOutOfSampleReturn;
    long double maxReturn = results_[0].averageOutOfSampleReturn;
    long double minStdDev = sqrt(results_[0].portfolioVariance);
    long double maxStdDev = sqrt(results_[0].portfolioVariance);
    
    for (const auto& result : results_) {
        long double ret = result.averageOutOfSampleReturn;
        long double std = sqrt(result.portfolioVariance);
        
        minReturn = min(minReturn, ret);
        maxReturn = max(maxReturn, ret);
        minStdDev = min(minStdDev, std);
        maxStdDev = max(maxStdDev, std);
    }
    
    // Create a simple scatter plot
    int plotHeight = 15;
    int plotWidth = 40;
    
    cout << fixed << setprecision(1);
    cout << "Return (%)" << endl;
    
    for (int row = plotHeight; row >= 0; row--) {
        long double currentReturn = minReturn + (maxReturn - minReturn) * row / plotHeight;
        
        // Print y-axis label
        if (row == plotHeight || row == 0 || row == plotHeight/2) {
            cout << setw(6) << currentReturn * 100 << " |";
        } else {
            cout << "       |";
        }
        
        // Plot points
        for (int col = 0; col <= plotWidth; col++) {
            long double currentStdDev = minStdDev + (maxStdDev - minStdDev) * col / plotWidth;
            
            bool hasPoint = false;
            for (const auto& result : results_) {
                long double ret = result.averageOutOfSampleReturn;
                long double std = sqrt(result.portfolioVariance);
                
                if (abs(ret - currentReturn) < (maxReturn - minReturn) / plotHeight * 0.8 &&
                    abs(std - currentStdDev) < (maxStdDev - minStdDev) / plotWidth * 0.8) {
                    hasPoint = true;
                    break;
                }
            }
            
            cout << (hasPoint ? "*" : " ");
        }
        cout << endl;
    }
    
    cout << "       |";
    for (int i = 0; i <= plotWidth; i++) cout << "_";
    cout << endl;
    
    cout << "       ";
    cout << setprecision(1) << minStdDev * 100;
    for (int i = 0; i < plotWidth - 8; i++) cout << " ";
    cout << maxStdDev * 100 << endl;
    cout << "                    Risk (Standard Deviation %)" << endl;
    cout << string(60, '=') << endl;
}

// Generate Python script for plotting
void Backtester::generatePythonPlotScript(const string& scriptName) const {
    ofstream script(scriptName);
    if (!script.is_open()) {
        cout << "Error: Could not create Python script " << scriptName << endl;
        return;
    }
    
    script << "#!/usr/bin/env python3" << endl;
    script << "import pandas as pd" << endl;
    script << "import matplotlib.pyplot as plt" << endl;
    script << "import numpy as np" << endl;
    script << endl;
    script << "print('Loading backtesting results for visualization...')" << endl;
    script << endl;
    script << "# Load data" << endl;
    script << "try:" << endl;
    script << "    ef_data = pd.read_csv('efficient_frontier.csv')" << endl;
    script << "    ta_data = pd.read_csv('target_vs_actual.csv')" << endl;
    script << "    results_data = pd.read_csv('backtesting_results.csv')" << endl;
    script << "    print(f'Loaded {len(results_data)} portfolio results')" << endl;
    script << "except FileNotFoundError as e:" << endl;
    script << "    print(f'Error: {e}')" << endl;
    script << "    print('Make sure CSV files are in the same directory as this script')" << endl;
    script << "    exit(1)" << endl;
    script << endl;
    script << "# Create figure with subplots" << endl;
    script << "fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))" << endl;
    script << "fig.suptitle('Portfolio Backtesting Analysis - Markowitz Model Performance', " << endl;
    script << "             fontsize=16, fontweight='bold')" << endl;
    script << endl;
    script << "# Plot 1: Efficient Frontier" << endl;
    script << "ax1.scatter(ef_data['StdDev'] * 100, ef_data['MeanReturn'] * 100, " << endl;
    script << "           alpha=0.8, s=60, c='blue', edgecolors='darkblue', linewidth=1)" << endl;
    script << "ax1.plot(ef_data['StdDev'] * 100, ef_data['MeanReturn'] * 100, " << endl;
    script << "         'b--', alpha=0.6, linewidth=2)" << endl;
    script << "ax1.set_xlabel('Risk (Standard Deviation %)', fontsize=12)" << endl;
    script << "ax1.set_ylabel('Expected Return (%)', fontsize=12)" << endl;
    script << "ax1.set_title('Efficient Frontier', fontsize=14, fontweight='bold')" << endl;
    script << "ax1.grid(True, alpha=0.3)" << endl;
    script << endl;
    script << "# Plot 2: Target vs Actual Returns" << endl;
    script << "ax2.scatter(ta_data['TargetReturn'] * 100, ta_data['ActualReturn'] * 100, " << endl;
    script << "           alpha=0.8, s=60, c='red', edgecolors='darkred', linewidth=1)" << endl;
    script << "# Add perfect prediction line" << endl;
    script << "min_val = min(ta_data['TargetReturn'].min(), ta_data['ActualReturn'].min()) * 100" << endl;
    script << "max_val = max(ta_data['TargetReturn'].max(), ta_data['ActualReturn'].max()) * 100" << endl;
    script << "ax2.plot([min_val, max_val], [min_val, max_val], 'k--', " << endl;
    script << "         label='Perfect Prediction', alpha=0.8, linewidth=2)" << endl;
    script << "ax2.set_xlabel('Target Return (%)', fontsize=12)" << endl;
    script << "ax2.set_ylabel('Realized Return (%)', fontsize=12)" << endl;
    script << "ax2.set_title('Target vs Realized Returns', fontsize=14, fontweight='bold')" << endl;
    script << "ax2.legend(fontsize=10)" << endl;
    script << "ax2.grid(True, alpha=0.3)" << endl;
    script << endl;
    script << "# Plot 3: Sharpe Ratios" << endl;
    script << "sharpe_ratios = results_data['ActualReturn'] / results_data['StdDev']" << endl;
    script << "colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sharpe_ratios)))" << endl;
    script << "bars = ax3.bar(range(len(results_data)), sharpe_ratios, " << endl;
    script << "               alpha=0.8, color=colors, edgecolor='black', linewidth=0.5)" << endl;
    script << "ax3.set_xlabel('Portfolio Index', fontsize=12)" << endl;
    script << "ax3.set_ylabel('Sharpe Ratio', fontsize=12)" << endl;
    script << "ax3.set_title('Sharpe Ratios by Portfolio', fontsize=14, fontweight='bold')" << endl;
    script << "ax3.grid(True, alpha=0.3, axis='y')" << endl;
    script << "# Add value labels on bars" << endl;
    script << "for i, (bar, sharpe) in enumerate(zip(bars, sharpe_ratios)):" << endl;
    script << "    if i % 4 == 0:  # Label every 4th bar to avoid crowding" << endl;
    script << "        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, " << endl;
    script << "                f'{sharpe:.2f}', ha='center', va='bottom', fontsize=8)" << endl;
    script << endl;
    script << "# Plot 4: Risk-Return Scatter with Color-coded Sharpe Ratios" << endl;
    script << "scatter = ax4.scatter(results_data['StdDev'] * 100, results_data['ActualReturn'] * 100, " << endl;
    script << "                     c=sharpe_ratios, s=80, alpha=0.8, cmap='RdYlGn', " << endl;
    script << "                     edgecolors='black', linewidth=0.5)" << endl;
    script << "ax4.set_xlabel('Risk (Standard Deviation %)', fontsize=12)" << endl;
    script << "ax4.set_ylabel('Realized Return (%)', fontsize=12)" << endl;
    script << "ax4.set_title('Risk-Return Profile (Color = Sharpe Ratio)', fontsize=14, fontweight='bold')" << endl;
    script << "ax4.grid(True, alpha=0.3)" << endl;
    script << "# Add colorbar" << endl;
    script << "cbar = plt.colorbar(scatter, ax=ax4)" << endl;
    script << "cbar.set_label('Sharpe Ratio', fontsize=10)" << endl;
    script << endl;
    script << "# Adjust layout and save" << endl;
    script << "plt.tight_layout()" << endl;
    script << "plt.savefig('portfolio_backtesting_analysis.png', dpi=300, bbox_inches='tight', " << endl;
    script << "            facecolor='white', edgecolor='none')" << endl;
    script << "plt.savefig('portfolio_backtesting_analysis.jpg', dpi=300, bbox_inches='tight', " << endl;
    script << "            facecolor='white', edgecolor='none')" << endl;
    script << "print('\\nPlots saved as:')" << endl;
    script << "print('- portfolio_backtesting_analysis.png')" << endl;
    script << "print('- portfolio_backtesting_analysis.jpg')" << endl;
    script << endl;
    script << "# Print summary statistics" << endl;
    script << "print('\\n=== SUMMARY STATISTICS ===')" << endl;
    script << "print(f'Number of portfolios analyzed: {len(results_data)}')" << endl;
    script << "print(f'Target return range: {ta_data[\"TargetReturn\"].min()*100:.1f}% to {ta_data[\"TargetReturn\"].max()*100:.1f}%')" << endl;
    script << "print(f'Realized return range: {results_data[\"ActualReturn\"].min()*100:.2f}% to {results_data[\"ActualReturn\"].max()*100:.2f}%')" << endl;
    script << "print(f'Average Sharpe ratio: {sharpe_ratios.mean():.3f}')" << endl;
    script << "print(f'Best Sharpe ratio: {sharpe_ratios.max():.3f} (Portfolio {sharpe_ratios.idxmax()})')" << endl;
    script << "print(f'Lowest risk portfolio: {results_data[\"StdDev\"].min()*100:.2f}% std dev (Portfolio {results_data[\"StdDev\"].idxmin()})')" << endl;
    script << endl;
    script << "plt.show()" << endl;
    
    script.close();
    cout << "Python plotting script generated: " << scriptName << endl;
    cout << "To create plots, run: python3 " << scriptName << endl;
}