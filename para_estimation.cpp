#include <iostream>
#include <cmath>
#include <vector>
#include "para_estimation.h"
#include "csv.h"
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <sstream>

using namespace std;
using Matrix2D = vector<vector<long double> >;
using Cube     = vector<Matrix2D>;  

                                                               
Matrix2D ParameterEstimator::rollingMeans (const vector<vector<long double> > &returns, int windowSize){
    cout << "Calculating rolling means with window size " << windowSize << "..." << endl;
    int row_comps = returns.size();
    int col_periods = returns[0].size();
    vector <vector <long double> > return_means(
        row_comps, //no. of rows
        vector<long double> (col_periods, 0.0) //for each row, stroe a vector with length col_periods, each filled with 0
    );

    for (int i = 0; i< row_comps; ++i){
        for (int j = 0; j<col_periods; ++j){
            if (j >= windowSize -1){
                long double sum = 0;
                for (int k = j - (windowSize - 1); k <= j; ++k){
                    sum += returns[i][k];
                }
                return_means[i][j] = sum/windowSize;
            }
        }
    }
    cout << "\nFinish running rollingMeans ....\n";
    return return_means;
};

Cube ParameterEstimator::rollingCovariance(
    const Matrix2D &returns, const Matrix2D &meanreturns,int windowSize){
        cout << "Calculating rolling covariance with window size " << windowSize << "..." << endl;
        int comps = returns.size();
        int periods = returns[0].size();
        Cube covariance_cube (periods, // rows
            Matrix2D(comps, vector <long double> (comps, 0.0))); //column becomes a 83 x 83 covariance matrix
        
        for (int p = windowSize - 1; p < periods; ++p){ //calendar index
            for (int comp1 = 0; comp1 < comps; ++comp1){
                for (int comp2 = 0; comp2< comps; ++comp2){
                    long double sum = 0;
                    for (int t = p - windowSize + 1; t<=p; ++t){
                        sum += ((returns[comp1][t] - meanreturns[comp1][p])* (returns[comp2][t] - meanreturns[comp2][p]));
                    }
                    covariance_cube[p][comp1][comp2] = sum / (windowSize -1);
                }
            }
        }
        cout << "\nFinish running rollingCovariance ....\n";
        return covariance_cube;
    };

double ParameterEstimator::string_to_double(const string& s )
{
	istringstream i(s);
	double x;
	if (!(i >> x))
		return 0;
	return x;
} 

void ParameterEstimator::readData(long double **data,const string& fileName)
{
	char tmp[20];
	ifstream file (strcpy(tmp, fileName.c_str()));
	Csv csv(file);
	string line;
	if (file.is_open())
	{
		int i=0;
		while (csv.getline(line) != 0) {
         	for (int j = 0; j < csv.getnfield(); j++)
            {
               long double temp=string_to_double(csv.getfield(j));
               cout << "Asset " << j << ", Return "<<i<<"="<< temp<<"\n"; //test
               data[j][i]=temp;
            }
            i++;
		}
		file.close();
	}
	else {cout <<fileName <<" missing\n";exit(0);}

    cout<<"Finish readData processing ----------";
}
     
