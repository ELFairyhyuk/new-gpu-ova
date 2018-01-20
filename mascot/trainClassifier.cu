/*
 * testTrainer.cpp
 *
 *  Created on: 31/10/2013
 *      Author: Zeyi Wen
 */

#include <sys/time.h>
#include "multiPredictor.h"
#include "trainClassifier.h"
#include "SVMCmdLineParser.h"
#include "classifierEvaluater.h"
#include "../svm-shared/Cache/cache.h"
#include "../svm-shared/HessianIO/deviceHessianOnFly.h"
#include "../SharedUtility/Timer.h"
#include "../SharedUtility/KeyValue.h"
#include "../SharedUtility/DataReader/LibsvmReaderSparse.h"

void trainSVM(SVMParam &param, string strTrainingFileName, int numFeature, SvmModel &model, bool evaluteTrainingError) {
    vector<vector<KeyValue> > v_v_Instance;
    vector<int> v_nLabel;

    int numInstance = 0;     //not used
    uint nNumofValue = 0;  //not used
    if(SVMCmdLineParser::numFeature > 0){
        numFeature = SVMCmdLineParser::numFeature;
    }
    else
        BaseLibSVMReader::GetDataInfo(strTrainingFileName, numFeature, numInstance, nNumofValue);
    LibSVMDataReader drHelper;
    drHelper.ReadLibSVMAsSparse(v_v_Instance, v_nLabel, strTrainingFileName, numFeature);

    SvmProblem problem(v_v_Instance, numFeature, v_nLabel);
//    problem = problem.getSubProblem(0,1);
    model.fit(problem, param);
    PRINT_TIME("training", trainingTimer)
    PRINT_TIME("working set selection",selectTimer)
    PRINT_TIME("pre-computation kernel",preComputeTimer)
    PRINT_TIME("iteration",iterationTimer)
    PRINT_TIME("g value updating",updateGTimer)
    model.saveLibModel(strTrainingFileName,problem);//save model in the same format as LIBSVM's
//    PRINT_TIME("2 instances selection",selectTimer)
//    PRINT_TIME("kernel calculation",calculateKernelTimer)
//    PRINT_TIME("alpha updating",updateAlphaTimer)
//    PRINT_TIME("init cache",initTimer)
    //evaluate training error
    if (evaluteTrainingError == true) {
        printf("Computing training accuracy...\n");
    //    evaluate(model, v_v_Instance, v_nLabel, ClassifierEvaluater::trainingError);//uncomment!!!!!!!!!!!!!!1
    }
}

void evaluateSVMClassifier(SvmModel &model, string strTrainingFileName, int numFeature) {
    cout<<" start decision value"<<endl;
    vector<vector<KeyValue> > v_v_Instance;
    vector<int> v_nLabel;

    int numInstance = 0;     //not used
    uint nNumofValue = 0;  //not used
    BaseLibSVMReader::GetDataInfo(strTrainingFileName, numFeature, numInstance, nNumofValue);
    LibSVMDataReader drHelper;
    drHelper.ReadLibSVMAsSparse(v_v_Instance, v_nLabel, strTrainingFileName, numFeature);
    //evaluate testing error
    //evaluate(model, v_v_Instance, v_nLabel, ClassifierEvaluater::testingError,ofs);//uncomment!!!!!!!!!!111
}

/**
 * @brief: evaluate the svm model, given some labeled instances.
 */
void evaluate(SvmModel &model, vector<vector<KeyValue> > &v_v_Instance, vector<int> &v_nLabel,
              vector<real> &classificationError, ofstream &ofs, real &pareto){
    int batchSize = 10000;

    //create a miss labeling matrix for measuring the sub-classifier errors.
    model.missLabellingMatrix = vector<vector<int> >(model.nrClass, vector<int>(model.nrClass, 0));
    bool bEvaluateSubClass = true; //choose whether to evaluate sub-classifiers
    if(model.nrClass == 2)  //absolutely not necessary to evaluate sub-classifers
        bEvaluateSubClass = false;

    MultiPredictor predictor(model, model.param);

    clock_t start, finish;
    start = clock();
    int begin = 0;
    vector<int> predictLabels;
    real minPosDecVal=1000;
    real minNegDecVal=1000;
    while (begin < v_v_Instance.size()) {
        //get a subset of instances
        int end = min(begin + batchSize, (int) v_v_Instance.size());
        vector<vector<KeyValue> > samples(v_v_Instance.begin() + begin,
                                          v_v_Instance.begin() + end);
        vector<int> vLabel(v_nLabel.begin() + begin, v_nLabel.begin() + end);
        if(bEvaluateSubClass == false)
            vLabel.clear();
        //predict labels for the subset of instances
    cout<<"in evaluate**************88"<<endl;
        vector<int> predictLabelPart = predictor.predict(samples, minPosDecVal,minNegDecVal, vLabel);
        predictLabels.insert(predictLabels.end(), predictLabelPart.begin(), predictLabelPart.end());
        begin += batchSize;
    }
    finish = clock();
    int numOfCorrect = 0;
    int neg=0;
    for (int i = 0; i < v_v_Instance.size(); ++i) {
    if(predictLabels[i]==1){
    neg++;
    }
        if (predictLabels[i] == v_nLabel[i])
            numOfCorrect++;
    }
    //pareto kl divergence
    pareto=5*fabs(log(minPosDecVal/minNegDecVal));

    printf("binary classifier accuracy = %.2f%%(%d/%d)\n", numOfCorrect / (float) v_v_Instance.size() * 100,
           numOfCorrect, (int) v_v_Instance.size());
    ofs<<"binary training accuracy: "<<numOfCorrect/(float) v_v_Instance.size()*100<<" ("<<numOfCorrect<<"/"<<v_v_Instance.size()<<")"<<"\n";
    printf("prediction time elapsed: %.2fs\n", (float) (finish - start) / CLOCKS_PER_SEC);

    if(bEvaluateSubClass == true){
        ClassifierEvaluater::evaluateSubClassifier(model.missLabellingMatrix, classificationError);
    }
}

float evaluateOVABinaryClassifier(vector<real>  &combDecValue, vector<vector<int> > &combPredictLabels, SvmModel &model, vector<vector<KeyValue> > &v_v_Instance, vector<int> &v_nLabel,
              vector<real> &classificationError){
    int batchSize = 10000;
    //create a miss labeling matrix for measuring the sub-classifier errors.
    model.missLabellingMatrix = vector<vector<int> >(model.nrClass, vector<int>(model.nrClass, 0));
    bool bEvaluateSubClass = true; //choose whether to evaluate sub-classifiers
    if(model.nrClass == 2)  //absolutely not necessary to evaluate sub-classifers
        bEvaluateSubClass = false;

    MultiPredictor predictor(model, model.param);

    clock_t start, finish;
    start = clock();
    int begin = 0;
    vector<int> predictLabels;
    while (begin < v_v_Instance.size()) {
        vector<real> decValue;
        //get a subset of instances
        int end = min(begin + batchSize, (int) v_v_Instance.size());
        vector<vector<KeyValue> > samples(v_v_Instance.begin() + begin,
                                          v_v_Instance.begin() + end);
        vector<int> vLabel(v_nLabel.begin() + begin, v_nLabel.begin() + end);
        if(bEvaluateSubClass == false)
            vLabel.clear();
        //predict labels for the subset of instances

       /* // procces for vote
    vector<int> predictLabelPart = predictor.predict(samples, vLabel);

        predictLabels.insert(predictLabels.end(), predictLabelPart.begin(), predictLabelPart.end());*/
        predictor.predictDecValue(decValue, samples);
        combDecValue.insert(combDecValue.end(), decValue.begin(), decValue.end());

        begin += batchSize;
    }
    finish = clock();
   /* process for vote 
    //combine bianry predictLabels
    combPredictLabels.push_back(predictLabels);*/
        
    if(bEvaluateSubClass == true){
        ClassifierEvaluater::evaluateSubClassifier(model.missLabellingMatrix, classificationError);
    }
    return (float) (finish - start) / CLOCKS_PER_SEC;
}


void evaluateOVAVote(vector<vector<KeyValue> > &testInstance, vector<int> &testLabel, vector<vector<int> > &combPredictLabels, vector<int> &originalPositiveLabel, float testingTime){    //read test set
      //vote for class
    int manyClassIns=0;//#instance that belong to more than one classes
    int NoClassIns=0;//#instance that does't belong to any class
    int correctIns=0;
    int nrClass=originalPositiveLabel.size();
    clock_t start,end;
    start=clock();
    for( int i=0;i<testInstance.size() ;i++){
        vector<int> vote(nrClass,0);
        int flag=0;
        int maxVote=0;
        for( int j=0;j<nrClass ;j++){
            if(combPredictLabels[j][i]==0)//if predictLabel=0 then instance belongs to the label 0 in jth bianrySVM
            {
        vote[j]++;
                flag++;
                maxVote=j;
            }
        }
            
        if(i<10)
            cout<<"flaglabel********"<<flag<<endl;
        if(flag==1){
            if(originalPositiveLabel[maxVote]==testLabel[i])
                correctIns++;
            cout<<"flag==1"<<endl;  
                }
        else if(flag>1){
            manyClassIns++;
            cout<<"many"<<endl;}
        else{
            NoClassIns++;
            cout<<"noclass"<<endl;
            }
        if(i<10)
            cout<<"manyclasslabel********"<<manyClassIns<<endl;
    }
    end=clock();
    testingTime+=(float)(end-start)/CLOCKS_PER_SEC;
    printf("classifier accuracy = %.2f%%(%d/%d)\n", correctIns / (float) testInstance.size() * 100,
           correctIns, (int) testInstance.size() );
    printf("number of unclaasifiable instances in OVA is %.2f%%(%d/%d)\n", manyClassIns / (float) testInstance.size() * 100, manyClassIns, testInstance.size());
    printf("number of NoClass instances in OVA is %.2f%%(%d/%d)\n", NoClassIns / (float) testInstance.size() * 100, NoClassIns, testInstance.size() );
    printf("prediction time elapsed: %.2fs\n",testingTime);

    
}

void evaluateOVADecValue(vector<vector<KeyValue> > &testInstance, vector<int> &testLabel, vector<vector<real> > &combDecValue, vector<int > originalPositiveLabel, float testingTime, ofstream &ofs){    //read test set
      //vote for class
    int manyClassIns=0;//#instance that belong to more than one classes
    int noClassIns=0;//#instance that does't belong to any class
    int correctIns=0;
    int iu_correctIns=0;//num of correct ins that calculated by ignoring unclassifiable data
    int nrClass=originalPositiveLabel.size();

    clock_t start,end;
    start=clock();
    for( int i=0;i<testInstance.size() ;i++){
        int flag=0;
		int manyClassFlag=0;
        int max=0;
        int iu_max=0;
        for( int j=0;j<nrClass ;j++){
            if(combDecValue[j][i]>=0&&combDecValue[j][i]>=combDecValue[max][i])//if predictLabel=0 then instance belongs to the label 0 in jth bianrySVM
            {
                flag++;
                max=j;
            }
            if(combDecValue[j][i]>0)
				manyClassFlag++;
            if(combDecValue[j][i]>combDecValue[iu_max][i])//ignore unclasiifiable ins
				iu_max=j;
        }
            
        if (originalPositiveLabel[iu_max]==testLabel[i])
        	iu_correctIns++;
		//new
		if(manyClassFlag>1)
			manyClassIns++;
		else if(flag>0){
            if (originalPositiveLabel[max]==testLabel[i])
				correctIns++;
		}
		//end new

       /**old
	   if(flag>0){
            int manyClassflag=0;
            for(int j=0;j<nrClass ;j++){
                if(j!=max){
                    if(combDecValue[j][i]==combDecValue[max][i]){
                        manyClassIns++;
                        manyClassflag++;
                        break;
                    }
                }
                
            }
            if(manyClassflag==0){
                if (originalPositiveLabel[max]==testLabel[i])
                    correctIns++;
            }
        }
		end old*/
        else{
            noClassIns++;
			}
    }
    end=clock();
    testingTime+=(float)(end-start)/CLOCKS_PER_SEC;
    printf("test  accuracy = %.2f%%(%d/%d)\n", correctIns / (float) testInstance.size() * 100,
           correctIns, (int) testInstance.size() );
    ofs<<"ignore unclassifiable ins test accuracy: "<<iu_correctIns / (float) testInstance.size()*100<<" ("<<iu_correctIns<<"/"<<testInstance.size()<<")"<<"\n";
    ofs<<"test accuracy: "<<correctIns / (float) testInstance.size()*100<<" ("<<correctIns<<"/"<<testInstance.size()<<")"<<"\n";
    
    printf("number of unclaasifiable instances in OVA is %.2f%%(%d/%d)\n", manyClassIns / (float) testInstance.size() * 100, manyClassIns, testInstance.size());
    ofs<<"test: # of manyClass instance: "<<manyClassIns / (float) testInstance.size()*100<<" ("<<manyClassIns<<"/"<<testInstance.size()<<")"<<"\n";
    
    printf("number of NoClass instances in OVA is %.2f%%(%d/%d)\n", noClassIns / (float) testInstance.size() * 100, noClassIns, testInstance.size() );
    ofs<<"test: #noClass instance:  "<<noClassIns / (float) testInstance.size()*100<<" ("<<noClassIns<<"/"<<testInstance.size()<<")"<<"\n";
    printf("prediction time elapsed: %.2fs\n",testingTime);
    ofs<<"testing time: "<<testingTime<<"\n";

}

float pearsonCorr(vector <KeyValue> x, vector< KeyValue> y, int numFeature){
	vector<float > deltaX;
	vector<float > deltaY;
	float avgX=0;
	float avgY=0;
	float sumxy=0;
	float sumx=0;
	float sumy=0;
	for(int i=0;i<x.size();i++){
		avgX+=x[i].featureValue;
	}
	avgX/=(float)numFeature;
	for(int i=0;i<y.size();i++){
		avgY+=y[i].featureValue;
	}
	avgY/=(float)numFeature;
	int kx=0;
	int ky=0;
	for(int i=0;i<numFeature;i++){
		if(kx<x.size()&&x[kx].id==i){
			deltaX.push_back(x[kx].featureValue-avgX);
			kx++;
			}
		else{
			deltaX.push_back(0-avgX);
		}
		if(ky<y.size()&&y[ky].id==i){
			deltaY.push_back(y[ky].featureValue-avgY);
			ky++;
			}
		else{
			deltaY.push_back(0-avgY);
		}
	}
	for(int i=0;i<numFeature;i++){
		sumxy+=deltaX[i]*deltaY[i];
		sumx+=deltaX[i]*deltaX[i];
		sumy+=deltaY[i]*deltaY[i];
	}
	float pearsonCorr=sumxy/(sqrt(sumx)*sqrt(sumy));
	return pearsonCorr;
}
float miniRatio(vector <KeyValue >x, vector<KeyValue > y, int numFeature){
	int kx=0;
	int ky=0;
	float sum=0;
	for(int i=0;i<numFeature;i++){
		if(kx<x.size()&&x[kx].id==i){
			if(ky<y.size()&&y[ky].id==i){
			float ratioxy=x[kx].featureValue/y[ky].featureValue;
			float ratioyx=y[ky].featureValue/x[kx].featureValue;
				if(ratioxy>ratioyx)
					sum+=ratioyx;
				else
					sum+=ratioxy;
			}
			kx++;
		}
		if(ky<y.size()&&y[ky].id==i)
			ky++;
	}
	return sum/numFeature;
}

void generateDomain(vector< vector<KeyValue> > &v_v_Instance, vector<int > &v_nLabel,int numFeature, int numInstance, int complexity,int nrClass, vector<int> imbl){
	vector<float> interval;
	float domain=1.0;
	for(int i=0;i<complexity;i++)
		domain*=2;
	for(int i=0;i<=domain;i++)
		interval.push_back(1.0*i/domain);
		for(int k=0;k<nrClass;k++){
			for(int i=0;i<domain;i+=nrClass){
				for(int j=0;j<numInstance*imbl[k];j++){//#instances in each interval
					vector<KeyValue > ins;
	    			for(int rt=0;rt<numFeature;rt++){
	        			srand((int)time(0)+rt);
	        			float rval=fmod((float)rand(),1.0/domain-0.00013);
						rval+=interval[k+i];
						KeyValue kv;
						kv.id=rt;
						kv.featureValue=rval;
						ins.push_back(kv);
					}
					v_v_Instance.push_back(ins);
					v_nLabel.push_back(k);
				}
	    
		}
		}
}

void trainOVASVM(SVMParam &param, string strTrainingFileName, int numFeature,  bool evaluteTrainingError, string strTestingFileName, ofstream &ofs) {
    //nrclass must >2
	/**
	//generate domain
    vector<vector<KeyValue> > v_v_Instance;
    vector<int> v_nLabel;
    vector<vector<KeyValue> > testInstance;
    vector<int> testLabel;
    int numInstance = 100;    
    int numTestInstance = 100;     //not used
	int nrClass=4;
	numFeature=10;
	int numTestFeature=1;
	int complexity=5;
	vector <int > imbl;
	imbl.push_back(1);
	imbl.push_back(1000);
	imbl.push_back(1000);
	imbl.push_back(1000);
	ofs<<"imbalance level "<<imbl[1]<<"\n";
	generateDomain(v_v_Instance,v_nLabel,numFeature, numInstance,complexity,nrClass,imbl);
	imbl[1]=1;
	imbl[2]=1;
	imbl[3]=1;
	generateDomain(testInstance,testLabel,numFeature, numTestInstance,complexity,nrClass,imbl);
    SvmProblem problem(v_v_Instance, numFeature, v_nLabel);
	ofs<<"complexity "<<complexity<<"\n";
	ofs<<"numInstance "<<numInstance<<"\n";

	//end generate domain
**/
//original
    vector<vector<KeyValue> > v_v_Instance;
    vector<int> v_nLabel;
    int numInstance = 0;     //not used
    uint nNumofValue = 0;  //not used
    if(SVMCmdLineParser::numFeature > 0){
        numFeature = SVMCmdLineParser::numFeature;
    }
    else
        BaseLibSVMReader::GetDataInfo(strTrainingFileName, numFeature, numInstance, nNumofValue);
    LibSVMDataReader drHelper;
    drHelper.ReadLibSVMAsSparse(v_v_Instance, v_nLabel, strTrainingFileName, numFeature);
    //build problem of all classes
    SvmProblem problem(v_v_Instance, numFeature, v_nLabel);
    int testNumInstance = 0;     //not used
    uint testNumofValue = 0;
	int numTestFeature=numFeature;
    vector<vector<KeyValue> > testInstance;
    vector<int> testLabel;
    BaseLibSVMReader::GetDataInfo(strTestingFileName, numTestFeature, testNumInstance, testNumofValue);
    LibSVMDataReader drHelper2;
    drHelper2.ReadLibSVMAsSparse(testInstance, testLabel, strTestingFileName, numTestFeature);
    
    int nrClass=problem.getNumOfClasses();
	//end origin
	
    //vector<SvmModel> combModel(nrClass);
    vector<vector<int> > combPredictLabels;//combine k binary predictLaebl
    vector<vector<real> > combDecValue(nrClass);
    vector<vector<int> > combTrainPredictLabels;//combine k binary predictLaebl
    vector<int> originalPositiveLabel(nrClass);
    float testingTime=0;
    float allTrainingTime=0;
    float avgTrainingTime=0;

	//claculate M(variance) M(aimilarity)
	vector<float > count(nrClass);
	float simi_gamma=1.0/100.0;
	float avg=0.0;
	float sum=0.0;
	float var=0.0;
	int numRandIns=1000;
	cout<<"feature "<<numFeature<<endl;
    for(int i=0;i<nrClass;i++){
		ofs<<"count in class "<<i<<" is "<<problem.count[i]<<"\n";
		sum+=problem.count[i];
		}
    for(int i=0;i<nrClass;i++){
	count[i]=(float)problem.count[i]/sum;
	avg+=count[i];
	}
	avg/=(float)nrClass;
    for(int i=0;i<nrClass;i++)
	var+=(count[i]-avg)*(count[i]-avg);
	var/=(float)(nrClass-1);
	cout<<"var "<<var<<endl;
	//M(similarity)
	vector<vector<int> > idxClass(nrClass);
	vector<vector<float> > v_v_simi;
	for(int j=0;j<nrClass;j++)
	    for(int rt=0;rt<numRandIns;rt++){
	        srand((int)time(0)+rt);
	        int ridx=rand()%(problem.count[j]-1);
	        idxClass[j].push_back(ridx);
	    }
	//calculate rbf kernel
	float simi=0.0;
	for(int i=0;i<nrClass;i++){
	    for(int j=i+1;j<nrClass;j++){
	        vector<float> v_simi;
		for(int n=0;n<numRandIns;n++){
	            vector<KeyValue> x;
	            vector<KeyValue> y;
		    x=v_v_Instance[problem.perm[problem.start[i]+idxClass[i][n]]];
		    y=v_v_Instance[problem.perm[problem.start[j]+idxClass[j][n]]];
		    //simi=expf(-simi_gamma * ( xi-xj)*(xi-xj));
		    int k=0;
		    int l=0;
		    for(int m=0;m<numFeature;m++){
			real xi=0.0;
			real yi=0.0;
		        if(k<x.size()&&x[k].id==m){
			    
		            xi=x[k].featureValue;
			    simi+=xi*xi;
			    if(l<y.size()&&m==y[l].id)    	    	
			        simi-=2*xi*y[l].featureValue;
		    	    k++;
			}
		        if(l<y.size()&&y[l].id==m){
		            yi=y[l].featureValue;    	    	
			    simi+=yi*yi;
		    	    l++;
			}
		    }  
		    simi=expf(-simi_gamma*simi);
		    v_simi.push_back(simi);
		}
		v_v_simi.push_back(v_simi);
		
	    }
	}
	vector<float> max_simi;//large kernel means closed
	for(int i=0;i<v_v_simi.size();i++){
	    float max_sim=v_v_simi[i][0];
	    for(int j=1;j<numRandIns;j++){
		max_sim=max_sim<v_v_simi[i][j]?v_v_simi[i][j]:max_sim;
	    }
	    max_simi.push_back(max_sim);
	}
	avg=0.0;
	for(int i=0;i<nrClass*(nrClass-1)/2;i++){
	//cout<<"iter= "<<i<<" max kernel "<<max_simi[i]<<endl;	
	    avg+=max_simi[i];
	}
	avg/=(nrClass*(nrClass-1)/2);
	cout<<"avg kernel"<<avg<<endl;
//	exit(1);
	//end rbf kernel
/*	//calculate Pearson Correlation
	float corrSum=0.0;
	for(int i=0;i<nrClass;i++){
	    for(int j=i+1;j<nrClass;j++){
			for(int n=0;n<numRandIns;n++){
	            vector<KeyValue> x;
	            vector<KeyValue> y;
		    	x=v_v_Instance[problem.perm[problem.start[i]+idxClass[i][n]]];
		    	y=v_v_Instance[problem.perm[problem.start[j]+idxClass[j][n]]];
				corrSum+=pearsonCorr(x,y,numFeature);
			}
		}
	}
	cout<<"Pearson Correlation is "<<corrSum*2/(nrClass*(nrClass-1)*numRandIns)<<"\n";
	//calculate minimum ratio
	float miniSum=0.0;
	for(int i=0;i<nrClass;i++){
	    for(int j=i+1;j<nrClass;j++){
			for(int n=0;n<numRandIns;n++){
	            vector<KeyValue> x;
	            vector<KeyValue> y;
		    	x=v_v_Instance[problem.perm[problem.start[i]+idxClass[i][n]]];
		    	y=v_v_Instance[problem.perm[problem.start[j]+idxClass[j][n]]];
				miniSum+=miniRatio(x,y,numFeature);
			}
		}
	}
	cout<<"minimum ratio is "<<miniSum*2/(nrClass*(nrClass-1)*numRandIns)<<"\n";
 	exit(1);
	//end measure
*/
    for(int i=0;i<nrClass;i++)
        originalPositiveLabel[i]=v_nLabel[problem.perm[problem.start[i]]];
/*    vector<SVMParam> v_param(nrClass,param); 
    v_param[0].C=4;
    v_param[0].gamma=0.5;
    v_param[1].C=1;
    v_param[1].gamma=0.000244;
    v_param[2].C=2;
    v_param[2].gamma=0.001952;
    v_param[3].C=1;
    v_param[3].gamma=0.25;
    v_param[4].C=8;
    v_param[4].gamma=0.000488;
    v_param[5].C=4;
    v_param[5].gamma=0.00000;
*/
    //train and predict bianry svm
	vector<real> pareto(nrClass,0);
    for(int i=0;i<nrClass;i++){
        v_v_Instance=problem.v_vSamples;
        v_nLabel=problem.v_nLabels;
    SvmModel model;
        //reassign the 0 and 1 label to instances.
        for(int m=0;m<problem.count[i];m++)
            v_nLabel[problem.perm[problem.start[i] + m]]=1;//0 denotes the positive class
        for(int n=0;n<nrClass;n++){
            if(n!=i){
                for(int l=0;l<problem.count[n];l++)
                    v_nLabel[problem.perm[problem.start[n] + l]]=-1;
            }
        }
    ofs<<"#i class instance "<<i<<": "<<problem.count[i]<<"\n";
        //for class i=0, in training phase, class i will be +1,
        //for other classes (i!=0), in training phase, class i will be -1
    //therfore, for class i!=0, swap the start[i] with the first instance in traing dataset, to make the class i to be the positive class in training phase
    if(i>0){
        vector<KeyValue> tempIns;
        int tempLabel;
    //swap label
    tempLabel=v_nLabel[problem.perm[problem.start[i]]];
        v_nLabel[problem.perm[problem.start[i]]]=v_nLabel[0];
        v_nLabel[0]=tempLabel;
    //swap instance 
        tempIns=v_v_Instance[problem.perm[problem.start[i]]];
        v_v_Instance[problem.perm[problem.start[i]]]=v_v_Instance[0];
        v_v_Instance[0]=tempIns;
      
    }
        //use instance with label 0 and 1 to build the problem
        SvmProblem binaryProblem(v_v_Instance, numFeature, v_nLabel);
        //problem.label=model.label  label[0]=the label of the first instance.

        model.fit(binaryProblem, param);//resize nrclass!!!!!solve->getsubproblem!!!!
        PRINT_TIME("training", trainingTimer)
        PRINT_TIME("working set selection",selectTimer)
        PRINT_TIME("pre-computation kernel",preComputeTimer)
        PRINT_TIME("iteration",iterationTimer)
        PRINT_TIME("g value updating",updateGTimer)
        model.saveLibModel(strTrainingFileName,problem);//save model in the same format as LIBSVM's
        allTrainingTime+=trainingTimer.getTotalTime();
        avgTrainingTime+=trainingTimer.getAverageTime();
//    PRINT_TIME("2 instances selection",selectTimer)
//    PRINT_TIME("kernel calculation",calculateKernelTimer)
//    PRINT_TIME("alpha updating",updateAlphaTimer)
//    PRINT_TIME("init cache",initTimer)

        //evaluate training error
        if (evaluteTrainingError == true) {
            printf("Computing training accuracy...\n");
                evaluate(model, v_v_Instance, v_nLabel, ClassifierEvaluater::trainingError, ofs, pareto[i]);
        }
        cout << "start binary test evaluation..." << endl;
//        testingTime+= evaluateOVABinaryClassifier(combDecValue[i], combPredictLabels, model, problem.v_vSamples, problem.v_nLabels, ClassifierEvaluater::testingError);
        testingTime+= evaluateOVABinaryClassifier(combDecValue[i], combPredictLabels, model, testInstance, testLabel, ClassifierEvaluater::testingError);
   
    }
	for(int n=1;n<nrClass;n++)
	    pareto[0]+=pareto[n];
	cout<<"pareto KL divegence "<<pareto[0]/nrClass<<endl;
    ofs<<"total training time "<<allTrainingTime<<" avg time"<<avgTrainingTime<<"\n";
    ofs<<"all evaluation"<<endl;
//    evaluateOVADecValue(problem.v_vSamples, problem.v_nLabels, combDecValue, originalPositiveLabel, testingTime, ofs);    //read train set
    evaluateOVADecValue(testInstance, testLabel, combDecValue, originalPositiveLabel, testingTime, ofs);    //read test set

    //evaluateOVAVote(testInstance, testLabel, combPredictLabels, originalPositiveLabel, testingTime);
    //evaluateOVA(v_v_Instance, v_nLabel, combTrainPredictLabels, originalPositiveLabel, testingTime);
}
