#ifndef MULTICLASSMETHODSELECT_H
#define MULTICLASSMETHODSELECT_H

#include "../SharedUtility/KeyValue.h"
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <sys/time.h>
#include "multiPredictor.h"
#include "trainClassifier.h"
#include "SVMCmdLineParser.h"
#include "classifierEvaluater.h"
#include "../svm-shared/Cache/cache.h"
#include "../svm-shared/HessianIO/deviceHessianOnFly.h"
#include "../SharedUtility/Timer.h"
#include "../SharedUtility/DataReader/LibsvmReaderSparse.h"
		      
real rbfKernel(vector<KeyValue> x, vector<KeyValue> y, float similarityGamma, int numFeature);
real calculateDis(vector<KeyValue> x, vector<KeyValue> y, float similarityGamma, int numFeature);
real calculateDis2Lp(vector<KeyValue> x, vector<KeyValue> cenA, vector<KeyValue> cenB);
void kmeans(int numCenter, vector<vector<KeyValue> > &centers, const vector<vector<KeyValue> > &vIns, int maxIter, int numFeature,float similarityGamma);
void kmeansPP(int numCenter, const vector<vector <KeyValue> > &vIns, vector <vector<KeyValue> > &centers, int numFeature,float similarityGamma);
int WillsonScore( float &lowwerboundA, float &lowwerboundB, float &lowwerboundAB, const vector<KeyValue> &cenA, const vector<KeyValue> &cenB, const vector<vector<KeyValue> > &vInsA, const vector<vector<KeyValue> > &vInsB, int numFeature,float similarityGamma);
float paretoKLDivergence(const vector<KeyValue> &bestA, const vector<KeyValue> &bestB, const vector<vector<KeyValue> > &vInsA, const vector<vector<KeyValue> > &vInsB, float similarityGamma,int numFeature,float &meanmad);
real xdoty(vector<KeyValue> x, vector<KeyValue> y, int numFeature);
real linearDis2Lp(vector<KeyValue> centerA, vector<KeyValue> centerB, vector<KeyValue> y, int numFeature);
int getLinearSimilarity(int numCenter, const vector<vector <KeyValue> > &vInsA,  const vector<vector <KeyValue> > &vInsB, int numFeature, float &similarity);
int getSimilarity(int numCenter, const vector<vector <KeyValue> > &vInsA,  const vector<vector <KeyValue> > &vInsB, int numFeature,float similarityGamma, float &similarity,float &meanmad);
void multiclassMethodSelection(SVMParam &param, string strTrainingFileName, int numFeature, ofstream &ofs);
#endif
