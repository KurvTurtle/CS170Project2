//Name: Christian Melendez
//SID: 862189972
//email: cmele014@ucr.edu
//Project 2: Feature Search

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
using namespace std;

//Create 2D Array from File (Return Features)
double** fileToFeaturesArray(int row, int col, string fileName) {
	//Initialize file
	ifstream testData(fileName);

	//2d dynamic array for features
	double** features = new double* [row + 1];
	for (int k = 0; k < row + 1; k++) {
		features[k] = new double[col];
	}

	//1d dynamic array for object classes
	double* objectClasses = new double[row + 1];

	//Write data into arrays
	for (int i = 0; i < row; i++) { //Rows
		for (int j = 0; j < col; j++) { //Coloumns
			if (j != 0) {
				testData >> features[i][j];
			}
			else {
				testData >> objectClasses[i];
			}
		}
	}
	return features;
}

//Create 2D Array from File (Return Object Classes)
double* fileToObjectClassesArray(int row, int col, string fileName) {
	//Initialize file
	ifstream testData(fileName);

	//2d dynamic array for features
	double** features = new double* [row + 1];
	for (int k = 0; k < row + 1; k++) {
		features[k] = new double[col];
	}

	//1d dynamic array for object classes
	double* objectClasses = new double[row + 1];

	//Write data into arrays
	for (int i = 0; i < row; i++) { //Rows
		for (int j = 0; j < col; j++) { //Coloumns
			if (j != 0) {
				testData >> features[i][j];
			}
			else {
				testData >> objectClasses[i];
			}
		}
	}
	return objectClasses;
}

//K-FOLD CROSS VALIDATION (LEAVE-ONE-OUT)
double kFoldCrossValidation(double* objectClasses, int* currentSet, int featureToAdd, int currentSetSize, double numObjects, int col, string fileName) {
	double** features = fileToFeaturesArray(numObjects, col, fileName);
	vector<int> nums{};

	//Vector of columns to erase from features array
	for (int i = 0; i < col - 1; i++) {
		nums.push_back(i + 1);
	}

	nums.erase(remove(nums.begin(), nums.end(), featureToAdd), nums.end()); //Keep featureToAdd

	for (int i = 0; i < currentSetSize; i++) { //Keep columns from currentSet
		nums.erase(remove(nums.begin(), nums.end(), currentSet[i]), nums.end());
	}
	
	//Erase the rest of the columns
	for(int j = 0; j < nums.size(); j++) {
		for (int i = 0; i < numObjects; i++) {
			features[i][nums[j]] = 0;
		}
	}

	//Find k nearest neighbors
	int nearestNeighborLocation, nearestNeighborClass;
	double nearestNeighborDist, distance, accuracy, sum;
	int correctlyClassified = 0;

	for (int i = 0; i < numObjects; i++) { //For each object (row)
		nearestNeighborDist = 999999999;
		for (int k = 1; k < numObjects; k++) { //For each object (row)
			if (k != i) { //Don't compare to yourself
				//Compute euclidean distance
				sum = 0;
				for (int j = 1; j < col + 1; j++) { //For each feature (1-10)
					sum += pow(features[i][j] - features[k][j], 2);
				}
				distance = sqrt(sum);
				
				//Update nearest neighbor information
				if (distance < nearestNeighborDist) {
					nearestNeighborDist = distance;
					nearestNeighborLocation = k;
					nearestNeighborClass = objectClasses[k];
				}
			}
		} //k loop

		//Increment if nearest neighbor correctly classified object
		if (objectClasses[i] == nearestNeighborClass) {
			correctlyClassified++;
		}
	} //i loop

	//Compute accuracy
	accuracy = correctlyClassified / numObjects;
	return accuracy;
}

//FEATURE SEARCH
void featureSearch(double** features, double* objectClasses, int size, double numObjects, string fileName) {
	//Dynamically sized empty array for current set of features
	int* setOfFeatures = new int[size];

	double accuracy, bestCurrentAccuracy; //From kFoldCrossValidation()
	double bestAccuracy = 0;
	int bestLevel = 0;

	for (int i = 1; i < size; i++) { //Levels
		//Print current level of search tree
		cout << "On the " << i;
		if (i == 1) {
			cout << "st ";
		} else if (i == 2) {
			cout << "nd ";
		} else if (i == 3) {
			cout << "rd ";
		} else {
			cout << "th ";
		}
		cout << "level of the search tree" << endl;

		bestCurrentAccuracy = 0;

		for (int k = 1; k < size; k++) { //Features
			//Only consider adding if not already added
			bool added = false;
			for (int j = 1; j < size; j++) {
				if (setOfFeatures[j] == k) {
					added = true;
				}
			}

			//Compute accuracy and consider adding feature k to level i
			
			if (!added) {
				accuracy = kFoldCrossValidation(objectClasses, setOfFeatures, k, i - 1, numObjects, size, fileName);
				cout << "Considering adding feature " << k << " with accuracy " << accuracy << endl;

				if (accuracy > bestCurrentAccuracy) {
					bestCurrentAccuracy = accuracy;
					setOfFeatures[i] = k;
					if (accuracy > bestAccuracy) {
						bestAccuracy = accuracy;
						bestLevel = i;
					}
				}
			}
		}

		//Print which feature was added to level i and the current set of features
		cout << "On level " << i << ", feature " << setOfFeatures[i] << " has been added to the current set: {";
		for (int n = 1; n < i; n++) {
			cout << setOfFeatures[n] << ", ";
		}
		cout << setOfFeatures[i] << "} w/ accuracy of " << bestCurrentAccuracy << endl << endl;
	}

	//Print out the best combination of features its accuracy
	cout << "BEST COMBINATION OF FEATURES: {";
	for (int n = 1; n < bestLevel; n++) {
		cout << setOfFeatures[n] << ", ";
	}
	cout << setOfFeatures[bestLevel] << "}" << endl;
	cout << "ACCURACY: " << bestAccuracy;
}

int main() {
	//Test Code
	/*double** features1 = fileToFeaturesArray(300, 11, "CS170_SMALLtestdata__16.txt");
	double* objectClasses1 = fileToObjectClassesArray(300, 11, "CS170_SMALLtestdata__16.txt");
	featureSearch(features1, objectClasses1, 11, 300, "CS170_SMALLtestdata__16.txt");*/

	double** features2 = fileToFeaturesArray(500, 101, "CS170_largetestdata__30.txt");
	double* objectClasses2 = fileToObjectClassesArray(500, 101, "CS170_largetestdata__30.txt");
	featureSearch(features2, objectClasses2, 101, 500, "CS170_largetestdata__30.txt");
	
	/*double** features3 = fileToFeaturesArray(300, 11, "CS170_small_special_testdata__95.txt");
	double* objectClasses3 = fileToObjectClassesArray(300, 11, "CS170_small_special_testdata__95.txt");
	featureSearch(features3, objectClasses3, 11, 300, "CS170_small_special_testdata__95.txt");*/

	return 0;
}