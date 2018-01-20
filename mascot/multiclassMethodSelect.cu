#include "multiclassMethodSelect.h"
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
#include <algorithm>
using std::cout;
using std::endl;
using std::vector;
using std::string;

real rbfKernel(vector<KeyValue> x, vector<KeyValue> y, float similarityGamma, int numFeature){
	int k=0;
	int l=0;
	real sum=0.0;
	for(int m=0;m<numFeature;m++){
		real xi=0.0;
		real yi=0.0;
		if(k<x.size()&&x[k].id==m){
			xi=x[k].featureValue;
			sum+=xi*xi;
			if(l<y.size()&&m==y[l].id)    	    	
				sum-=2*xi*y[l].featureValue;
		    k++;
		}
		if(l<y.size()&&y[l].id==m){
		    yi=y[l].featureValue;    	    	
			sum+=yi*yi;
		    l++;
		}
	}
//	cout<<"sum"<<sum<<endl;
	return exp(-similarityGamma*sum);
}

real calculateDis(vector<KeyValue> x, vector<KeyValue> y, float similarityGamma, int numFeature){
	real sqDis=2-2*rbfKernel(x,y,similarityGamma,numFeature);//dis^2=k(x,x)-2k(x,y)+k(y,y)
	return sqrt(sqDis);
}

real calculateDis2Lp(vector<KeyValue> x, vector<KeyValue> cenA, vector<KeyValue> cenB,float similarityGamma,int numFeature){
	real xA=rbfKernel(x,cenA,similarityGamma,numFeature);
	real xB=rbfKernel(x,cenB,similarityGamma,numFeature);
	real disAB=calculateDis(cenA,cenB,similarityGamma,numFeature);
//	cout<<"disAb "<<xA<<endl;
	return (xA-xB)/disAB;//disAB!=0
}

void kmeans(int numCenter, vector<vector<KeyValue> > &centers, const vector<vector<KeyValue> > &vIns, int maxIter, int numFeature,float similarityGamma){
	int iter=0;
	vector<vector<int> > idx(numCenter);
	while(iter<maxIter){
		//distance from instance to k centers
		for(int i=0;i<vIns.size();i++){
			float minDis=calculateDis(vIns[i],centers[0],similarityGamma,numFeature);
			int minCenter=0;
			for(int j=1;j<numCenter;j++){
				float dis=calculateDis(vIns[i],centers[j],similarityGamma,numFeature);
				if(dis<minDis){
					minDis=dis;		//useless
					minCenter=j;
				}
			}
			idx[minCenter].push_back(i);
		}
		//calculate the new center
		for(int i=0;i<numCenter;i++){
			//id=index of avg(except for index 0)
			vector<real> avg(numFeature+1,0);
			for(int j=0;j<idx[i].size();j++){
				for(int k=0;k<vIns[idx[i][j]].size();k++){
					avg[vIns[idx[i][j]][k].id]+=vIns[idx[i][j]][k].featureValue;
				}
			}
			//replace centers[i] with new center
			centers[i].clear();
			for(int j=1;j<=numFeature;j++){
				if(avg[j]!=0){
					avg[j]/=idx[i].size();
					KeyValue kv;
					kv.id=j;
					kv.featureValue=avg[j];
					centers[i].push_back(kv);
				}
			}
		}
		iter++;
	}//end while
	//within class unbalance in class B idx[].size
}

/*generate k centers in one class*/
void kmeansPP(int numCenter, const vector<vector <KeyValue> > &vIns, vector <vector<KeyValue> > &centers, int numFeature,float similarityGamma){
	vector<vector<KeyValue> > ins(vIns);
	centers.push_back(ins[0]);//initial a random center
	ins.erase(ins.begin());
	for(int i=1;i<numCenter;i++){
		vector<float > dis(ins.size());
		for(int j=0;j<ins.size();j++){
			float minDis=1000;//??????????????????????
			for(int k=0;k<i;k++){
				float fdis=calculateDis(centers[k],ins[j],similarityGamma,numFeature);
			    minDis=fdis<minDis?fdis:minDis;		
			}
			dis[j]=minDis;
		}
		float sum=0;
		for(int j=0;j<ins.size();j++)
			sum+=dis[j]*dis[j];
		for(int j=0;j<ins.size();j++)
			dis[j]=dis[j]*dis[j]/sum;
		//random index
		srand(time(0)+i);
		float randNum=(float)rand()/(float)RAND_MAX;//rand%1001.0)/1000.0
		int j;
		for(j=0;j<ins.size();j++){
			if(randNum<dis[j])
				break;
			else
				randNum-=dis[j];
		}
		//choose a new center
		centers.push_back(ins[j]);
		ins.erase(ins.begin()+j);
	}
	int maxIter=2;
	kmeans(numCenter,centers,vIns,maxIter,numFeature,similarityGamma);
}


int WillsonScore( float &lowwerboundA, float &lowwerboundB, float &lowwerboundAB, const vector<KeyValue> &cenA, const vector<KeyValue> &cenB, const vector<vector<KeyValue> > &vInsA, const vector<vector<KeyValue> > &vInsB, int numFeature,float similarityGamma){
	//get distance from center A to Lp, d(A,Lp). The distance between ins from cluster A and Lp, d(ins,Lp), should have the same label as d(A,Lp)
	float disA=calculateDis2Lp(cenA,cenA,cenB,similarityGamma,numFeature);
	float disB=calculateDis2Lp(cenB,cenA,cenB,similarityGamma,numFeature);
	int errA=0;
	int errB=0;

	for(int i=0;i<vInsA.size();i++){
		float temp=calculateDis2Lp(vInsA[i],cenA,cenB,similarityGamma,numFeature);
		if(temp*disA<0)//&&fabs(temp)>0.01)
		errA++;
	}
	for(int i=0;i<vInsB.size();i++){
	        float temp=calculateDis2Lp(vInsB[i],cenA,cenB,similarityGamma,numFeature);
		if(temp*disB<0)
			errB++;
	}
	
	int nA=vInsA.size();
	int nB=vInsB.size();
	lowwerboundA=(1/(nA+1.96*1.96))*(nA-errA+0.5*1.96*1.96-1.96*sqrt(errA*(nA-errA)/nA+1.96*1.96/4)); 
	lowwerboundB=(1/(nB+1.96*1.96))*(nB-errB+0.5*1.96*1.96-1.96*sqrt(errB*(nB-errB)/nB+1.96*1.96/4));
	lowwerboundAB=(1/(nA+nB+1.96*1.96))*(nA-errA+nB-errB+0.5*1.96*1.96-1.96*sqrt((errA+errB)*(nA-errA+nB-errB)/(nA+nB)+1.96*1.96/4));//if the score is higer, then (nA-errA) is higher, accuracy is higher
	return errA+errB;
	//cout<<"error A"<<errA<<" err B "<<errB<<endl;
	//cout<<"error AB"<<errA+errB<<endl;
}

float paretoKLDivergence(const vector<KeyValue> &bestA, const vector<KeyValue> &bestB, const vector<vector<KeyValue> > &vInsA, const vector<vector<KeyValue> > &vInsB, float similarityGamma,int numFeature,float &meanmad){
	int a=5;//a>2
	float minDisA=calculateDis2Lp(vInsA[0],bestA,bestB,similarityGamma,numFeature);
	float sumA=minDisA;
	float madA=0;
	float madB=0;
	vector<float> disA;
	vector<float> disB;
	for(int i=1;i<vInsA.size();i++){
		disA.push_back(calculateDis2Lp(vInsA[i],bestA,bestB,similarityGamma,numFeature));
		sumA+=disA[i];
		minDisA=minDisA>disA[i]?disA[i]:minDisA;
	}
	float minDisB=calculateDis2Lp(vInsB[0],bestA,bestB,similarityGamma,numFeature);
	float sumB=minDisB;
	for(int i=1;i<vInsB.size();i++){
		disB.push_back(calculateDis2Lp(vInsB[i],bestA,bestB,similarityGamma,numFeature));
		sumB+=disB[i];
		//cout<<"dis "<<dis<<endl;
		minDisB=minDisB>disB[i]?disB[i]:minDisB;
	}
	sumA/=vInsA.size();
	sumB/=vInsB.size();
	for(int i=0;i<disA.size();i++){
	    madA+=abs(disA[i]-sumA);
	}
	for(int i=0;i<disB.size();i++){
	    madB+=abs(disB[i]-sumB);
	}
	madA/=disA.size();
	madB/=disB.size();
	if(madA>=madB){
		meanmad=abs(sumA-sumB)/madA;
//		cout<<"largest mad "<<madA<<endl;
		cout<<"mean difference / mad "<<abs(sumA-sumB)/madA<<endl;
	}
	else{
		meanmad=abs(sumA-sumB)/madB;
//		cout<<"largest mad "<<madB<<endl;
		cout<<"mean difference / mad "<<abs(sumA-sumB)/madB<<endl;
	}
		//cout<<"minA "<<minDisA<<" "<<minDisB<<" "<<log(minDisA/minDisB)<<endl;
	if(minDisA*minDisB<=0){
		return 0;
	}
	else{
		//minDisB can not be 0, minDisA/minDisB should >0
		return fabs(a*log(minDisA/minDisB));
	
	}
}
float gaussianKLDivergence(const vector<KeyValue> &bestA, const vector<KeyValue> &bestB, const vector<vector<KeyValue> > &vInsA, const vector<vector<KeyValue> > &vInsB, float similarityGamma,int numFeature,float &errDisAll){
	float minDisA=calculateDis2Lp(vInsA[0],bestA,bestB,similarityGamma,numFeature);
	float sumA=0;
	float sumB=0;
	float varA=0;
	float varB=0;
	vector<float> v_disA;
	vector<float> v_disB;
	//cal dis that missclassified
	float DisCenA=calculateDis2Lp(bestA,bestA,bestB,similarityGamma,numFeature);
	float DisCenB=calculateDis2Lp(bestB,bestA,bestB,similarityGamma,numFeature);
	int count=0;
	float errDis=0;
	for(int i=0;i<vInsA.size();i++){
		float temp=calculateDis2Lp(vInsA[i],bestA,bestB,similarityGamma,numFeature);
		if(temp*DisCenA<0){
			errDis+=fabs(temp);
			count++;
			//if true, ins[i] will not be inclueded in the distribution
//			if(fabs(temp)<0.5*fabs(DisCenB)){
				v_disA.push_back(temp);
				sumA+=v_disA[i];
//		}
	}
		else{
			v_disA.push_back(temp);
			sumA+=v_disA[i];
		}
}
	float avgA=sumA/v_disA.size();
	for(int i=0;i<v_disA.size();i++){
		varA+=(v_disA[i]-avgA)*(v_disA[i]-avgA);
	}
	if(v_disA.size()>1)
		varA/=(v_disA.size()-1);
	//for classB

	for(int i=0;i<vInsB.size();i++){
		float temp=calculateDis2Lp(vInsB[i],bestA,bestB,similarityGamma,numFeature);
		if(temp*DisCenB<0){
			errDis+=fabs(temp);
			count++;
//			if(fabs(temp)<0.5*fabs(DisCenA)){
				v_disB.push_back(temp);
				sumB+=v_disB[i];
//		}
	}
		else{
			v_disB.push_back(temp);
			sumB+=v_disB[i];
		}
	}

	errDisAll=errDis/count;
	float avgB=sumB/v_disB.size();
	for(int i=0;i<v_disB.size();i++)
		varB+=(v_disB[i]-avgB)*(v_disB[i]-avgB);
	if(v_disB.size()>1)
		varB/=(v_disB.size()-1);
	if(varA!=0&&varB!=0){// in case that denominator =0
		float kl=fabs(log(sqrt(varB)/sqrt(varA))+(varA+(avgA-avgB)*(avgA-avgB))/(2*varB)-0.5)+fabs(log(sqrt(varA)/sqrt(varB))+(varB+(avgA-avgB)*(avgA-avgB))/(2*varA)-0.5);
		return kl/2;  //log=ln in c++
//		return fabs(log(sqrt(varB)/sqrt(varA))+(varA+(avgA-avgB)*(avgA-avgB))/(2*varB)-0.5);
	}
	else
		return LONG_MAX;//max 
}


int getSimilarity(int numCenter, const vector<vector <KeyValue> > &vInsA,  const vector<vector <KeyValue> > &vInsB, int numFeature,float similarityGamma, float &similarity,float &errDisAll){
		//select numCenter centers for class A
		vector<vector<KeyValue> > centersA;
		kmeansPP(numCenter, vInsA, centersA, numFeature, similarityGamma);
		//select numCenter centers for class B
		vector<vector<KeyValue> > centersB;
		kmeansPP(numCenter, vInsB, centersB, numFeature, similarityGamma);
		
		//find the best Lp between the best center A and best center B
		vector<KeyValue> bestA(centersA[0]);
		vector<KeyValue> bestB(centersB[0]);
		float willsonscore=0;
		float tempA,tempB;
		int bestErrAB=WillsonScore( tempA, tempB, willsonscore, centersA[0],centersB[0], vInsA, vInsB,numFeature,similarityGamma);
		for (int m=0;m<numCenter;m++){
			for (int n=0;n<numCenter;n++){
				float lowwerboundA=0;
				float lowwerboundB=0;
				float lowwerboundAB=0;
				int errAB=WillsonScore( lowwerboundA, lowwerboundB, lowwerboundAB, centersA[m],centersB[n], vInsA, vInsB,numFeature,similarityGamma);
				if(lowwerboundAB>willsonscore){
					willsonscore=lowwerboundAB;
					bestErrAB=errAB;
					bestA=centersA[m];
					bestB=centersB[n];
				}
			}
		}
		//calculate the simlarity of instances besides the best Lp 
	similarity=gaussianKLDivergence(bestA,bestB, vInsA, vInsB, similarityGamma, numFeature,errDisAll);
	
	return bestErrAB;
}

real xdoty(vector<KeyValue> x, vector<KeyValue> y, int numFeature){
	int k=0;
	int l=0;
	float sum=0.0;
	for(int m=0;m<numFeature;m++){
		real xi=0.0;
		if(k<x.size()&&x[k].id==m){
			xi=x[k].featureValue;
			if(l<y.size()&&m==y[l].id)    	    	
				sum+=xi*y[l].featureValue;
		    k++;
		}
		if(l<y.size()&&y[l].id==m){
		    l++;
		}
	}
	return sum;
}
real linearDis2Lp(vector<KeyValue> centerA, vector<KeyValue> centerB, vector<KeyValue> y, int numFeature){

	float xAy=xdoty(centerA,y,numFeature);
	float xBy=xdoty(centerB,y,numFeature);
	float xA=xdoty(centerA,centerA,numFeature);
	float xB=xdoty(centerB,centerB,numFeature);
	float xAxB=xdoty(centerA,centerB,numFeature);
	float dis=2*xAy-2*xBy-xA+xB;
	return dis/sqrt(xA+xB-2*xAxB);
}
int getLinearSimilarity(int numCenter, const vector<vector <KeyValue> > &vInsA,  const vector<vector <KeyValue> > &vInsB, int numFeature, float &similarity){

		vector <KeyValue> centerA;
		vector <KeyValue> centerB;
		for(int i=0;i<numCenter;i++){
			//id=index of avg(except for index 0)
			vector<real> avgA(numFeature+1,0);
			for(int j=0;j<vInsA.size();j++){
				for(int k=0;k<vInsA[j].size();k++){
					avgA[vInsA[j][k].id]+=vInsA[j][k].featureValue;
				}
			}
			for(int j=1;j<=numFeature;j++){
				if(avgA[j]!=0){
					avgA[j]/=vInsA.size();
					KeyValue kv;
					kv.id=j;
					kv.featureValue=avgA[j];
					centerA.push_back(kv);
				}
			}
			vector<real> avgB(numFeature+1,0);
			for(int j=0;j<vInsB.size();j++){
				for(int k=0;k<vInsB[j].size();k++){
					avgB[vInsB[j][k].id]+=vInsB[j][k].featureValue;
				}
			}
			for(int j=1;j<=numFeature;j++){
				if(avgB[j]!=0){
					avgB[j]/=vInsB.size();
					KeyValue kv;
					kv.id=j;
					kv.featureValue=avgB[j];
					centerB.push_back(kv);
				}
			}
	}

	vector<float > disA;
	vector<float > disB;
	float avgA=0.0;
	float avgB=0.0;
	float varA=0.0;
	float varB=0.0;
	int err=0;
	float disCenA=linearDis2Lp(centerA,centerA,centerB,numFeature);
	float disCenB=linearDis2Lp(centerB,centerA,centerB,numFeature);
	for(int i=0;i<vInsA.size();i++){
		disA.push_back(linearDis2Lp(vInsA[i],centerA,centerB,numFeature));
		avgA+=disA[i];
		if(disA[i]*disCenA<0)
			err++;
	}
	avgA/=vInsA.size();
	for(int i=0;i<disA.size();i++){
		varA+=(disA[i]-avgA)*(disA[i]-avgA);
	}
	if(disA.size()>1)
		varA/=(disA.size()-1);

	for(int i=0;i<vInsB.size();i++){
		disB.push_back(linearDis2Lp(vInsB[i],centerA,centerB,numFeature));
		avgB+=disB[i];
		if(disB[i]*disCenB<0)
			err++;
	}
	avgB/=vInsB.size();
	for(int i=0;i<disB.size();i++){
		varB+=(disB[i]-avgB)*(disB[i]-avgB);
	}
	if(disB.size()>1)
		varB/=(disB.size()-1);

	if(varA!=0&&varB!=0){// in case that denominator =0
		float kl=fabs(log(sqrt(varB)/sqrt(varA))+(varA+(avgA-avgB)*(avgA-avgB))/(2*varB)-0.5)+fabs(log(sqrt(varA)/sqrt(varB))+(varB+(avgA-avgB)*(avgA-avgB))/(2*varA)-0.5);
		similarity=kl/2;
//		similarity=fabs(log(sqrt(varB)/sqrt(varA))+(varA+(avgA-avgB)*(avgA-avgB))/(2*varB)-0.5);
	}
	else
		similarity= LONG_MAX;//max 

	return err;
}


int getWithinClass(int numCenter, const vector<vector <KeyValue> > &vInsA,  const vector<vector <KeyValue> > &vInsB, int numFeature,float similarityGamma, float &avgA, float &avgB){
		//select numCenter centers for class A
		vector<vector<KeyValue> > centersA;
		kmeansPP(numCenter, vInsA, centersA, numFeature, similarityGamma);
		//select numCenter centers for class B
		vector<vector<KeyValue> > centersB;
		kmeansPP(numCenter, vInsB, centersB, numFeature, similarityGamma);
	float sumDis=0;
	for(int i=0;i<numCenter;i++){
		for(int j=i+1;j<numCenter;j++){
			sumDis+=calculateDis(centersA[i],centersA[j],similarityGamma,numFeature);
		}
	}
	avgA=sumDis*2/((numCenter-1)*numCenter);
	sumDis=0;
	for(int i=0;i<numCenter;i++){
		for(int j=i+1;j<numCenter;j++){
			sumDis+=calculateDis(centersB[i],centersB[j],similarityGamma,numFeature);
		}
	}
	avgB=sumDis*2/((numCenter-1)*numCenter);
}

void checkRestCalsses(SvmProblem &problem, vector<vector<KeyValue> > &v_v_Instance,vector<int> &v_nLabel, int numClasses, float similarityGamma, int numFeature){
	int numCenter=1;
	int countPos=0;
	int countNeg=0;
	int flag=0;
	vector<vector<KeyValue> > centers;
	for (int i=0;i<numClasses;i++){
	//select numCenter centers for class A
		vector<vector<KeyValue> > vInsA;
		for(int j=0;j<v_nLabel.size();j++){
			if(problem.label[i]==v_nLabel[j]){
				vInsA.push_back(v_v_Instance[i]);
			}
		}
		vector<vector<KeyValue> > centersA;
		kmeansPP(numCenter, vInsA, centersA, numFeature, similarityGamma);
		centers.push_back(centersA[0]);

	}
	for(int i=0;i<numClasses;i++){
		for(int j=i+1;j<numClasses;j++){
		    countPos=0;
		    countNeg=0;
		    for(int m=0;m<numClasses;m++){
			if(m!=i&&m!=j){
			if(calculateDis2Lp(centers[m],centers[i],centers[j],similarityGamma,numFeature)>0)
			    countPos++;
			else 
			    countNeg++;}
		    }
		    if(countPos>countNeg){
			if(countNeg>numClasses*0.2)
			    flag++;
		    }
		    else{
			if(countPos>numClasses*0.2)
			    flag++;
		    }
		}
	}
	if(flag>((numClasses-1)*numClasses/4))//if flag >#classifiers/2, then use ova
		cout<<"should use ova "<<flag<<" "<<numClasses<<endl;	
	else
		cout<<"should use ovo "<<flag<<" "<<numClasses<<endl;	
}

/*main*/
void multiclassMethodSelection(SVMParam &param, string strTrainingFileName, int numFeature, ofstream &ofs){
	vector<vector<KeyValue> > v_v_Instance;
	vector<int> v_nLabel;
	int numInstance=0;
	unsigned int nNumofValue=0;
	if(SVMCmdLineParser::numFeature > 0){
 		numFeature = SVMCmdLineParser::numFeature; 
	}
 	BaseLibSVMReader::GetDataInfo(strTrainingFileName, numFeature, numInstance, nNumofValue);
 	LibSVMDataReader drHelper;
	drHelper.ReadLibSVMAsSparse(v_v_Instance, v_nLabel, strTrainingFileName, numFeature);
	SvmProblem problem(v_v_Instance, numFeature, v_nLabel);
	int nrClass=problem.getNumOfClasses();
//varinace
/**
	float classVar=0;
	float classAvg=0;
	for(int i =0;i<nrClass;i++){
		classAvg+=problem.count[i];
}	
	classAvg/=nrClass;
	for(int i=0;i<nrClass;i++)
		classVar+=(problem.count[i]-classAvg)*(problem.count[i]-classAvg);
        cout<<"ovo var "<<classVar/(nrClass-1)<<endl;
*******/

	int numCenter=1;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	float similarityGamma=param.gamma;//1.0/(float)numFeature;//param.gamma;
	//OVA
	vector<float> errAB(nrClass);
	vector<float> similarity(nrClass,0);
	vector<float> errDisAll(nrClass,0);
	for(int classA=0;classA<nrClass;classA++){
			cout<<" label**** "<<problem.label[classA]<<endl;
			cout<<" count**** "<<problem.count[classA]<<endl;
		vector<vector<KeyValue> > vInsA;//instances from class A
		vector<vector<KeyValue> > vInsB;//instances from class B (the rest ins)
		for(int i=0;i<v_nLabel.size();i++){
			if(problem.label[classA]==v_nLabel[i]){
				vInsA.push_back(v_v_Instance[i]);
			}
			else{
				vInsB.push_back(v_v_Instance[i]);
		}}
		errAB[classA]= (float)getLinearSimilarity(numCenter, vInsA,vInsB, numFeature,similarity[classA]);
	//	errAB[classA]=(float)getSimilarity(numCenter, vInsA,vInsB, numFeature, similarityGamma,similarity[classA],errDisAll[classA]);
		//getWithinClass(numCenter, vInsA,vInsB, numFeature, similarityGamma,avgA,avgB);
		errAB[classA]/=v_nLabel.size();
	}

	for(int i=1;i<nrClass;i++)
	    errDisAll[0]+=errDisAll[i];
	cout<<"OVA avg errDis is "<<errDisAll[0]/nrClass<<endl;
//	for(int i=1;i<nrClass;i++)
//		similarity[0]+=similarity[i];
	std::sort(similarity.begin(),similarity.end());//<
	std::sort(errAB.begin(),errAB.end());
	float avg=0;
	float avgErrAB=0;
/**
	for(int i=1;i<nrClass-1;i++){
		avg+=similarity[i];
		avgErrAB+=errAB[i];
	}
	avg/=(nrClass-2);
	avgErrAB/=(nrClass-2);
*****/

	for(int i=0;i<nrClass;i++){
		avg+=similarity[i];
		avgErrAB+=errAB[i];
	}
	avg/=(nrClass);
	avgErrAB/=(nrClass);

	float var=0;
	for(int i=0;i<nrClass;i++)
		var+=(similarity[i]-avg)*(similarity[i]-avg);
	var/=nrClass;
	cout<<strTrainingFileName<<endl;
	cout<<"errAB= "<<avgErrAB<<endl;
	cout<<"OVA: Pareto KL divergence similarity is avg= "<<avg<<endl;
	cout<<"OVA: Pareto KL divergence similarity is var= "<<var<<endl;
	//OVO

	vector<float> errABOVO(nrClass*(nrClass-1)/2);
	vector<float> similarityOVO((nrClass-1)*nrClass/2,0);
	vector<float> errDisAllOVO((nrClass-1)*nrClass/2,0);
	for(int classA=0;classA<nrClass;classA++){
		for(int classB=classA+1;classB<nrClass;classB++){
		vector<vector<KeyValue> > vInsA;
		vector<vector<KeyValue> > vInsB;
		for(int i=0;i<v_nLabel.size();i++){
			if(problem.label[classA]==v_nLabel[i]){
				vInsA.push_back(v_v_Instance[i]);
			}
			else if(problem.label[classB]==v_nLabel[i]){
				vInsB.push_back(v_v_Instance[i]);
			}
		}
		//cout<<"test"<<endl;
	errABOVO[classA*nrClass-classA*classA/2-3*classA/2+classB-1]=(float)getLinearSimilarity(numCenter, vInsA,vInsB, numFeature,similarityOVO[classA*nrClass-classA*classA/2-3*classA/2+classB-1]);
	//errABOVO[classA*nrClass-classA*classA/2-3*classA/2+classB-1]=(float)getSimilarity(numCenter, vInsA,vInsB, numFeature, similarityGamma,similarityOVO[classA*nrClass-classA*classA/2-3*classA/2+classB-1],errDisAllOVO[classA*nrClass-classA*classA/2-3*classA/2+classB-1]);
//	cout<<"similarity= "<<similarityOVO[classA*nrClass-classA*classA/2-3*classA/2+classB-1]<<endl;
		errABOVO[classA*nrClass-classA*classA/2-3*classA/2+classB-1]/=(problem.count[classA]+problem.count[classB]);
		}
	}
	avg=0;
	avgErrAB=0;
/***	
	std::sort(similarityOVO.begin(),similarityOVO.end());//<
	std::sort(errABOVO.begin(),errABOVO.end());
	for(int i=1;i<nrClass*(nrClass-1)/2-1;i++){
		avg+=similarityOVO[i];
		avgErrAB+=errABOVO[i];
	}
	avg/=(nrClass*(nrClass-1)/2-2);
	avgErrAB/=(nrClass*(nrClass-1)/2-2);
*******/
	for(int i=0;i<nrClass*(nrClass-1)/2;i++){
		avg+=similarityOVO[i];
		avgErrAB+=errABOVO[i];
	}
	avg/=(nrClass*(nrClass-1)/2);
	avgErrAB/=(nrClass*(nrClass-1)/2);

	var=0;
	for(int i=0;i<nrClass*(nrClass-1)/2;i++)
		var+=(similarityOVO[i]-avg)*(similarityOVO[i]-avg);
	var/=nrClass*(nrClass-1);
	var*=2;
//	int count=0;
	for(int i=1;i<nrClass*(nrClass-1)/2;i++)
	    errDisAll[0]+=errDisAll[i];
	cout<<"OVO avg errDis is "<<errDisAll[0]*2/nrClass*(nrClass-1)<<endl;
//	    cout<<"errABOVO "<<errABOVO[i]<<endl;
//		if(similarityOVO[i]==0)
//			count++;
//		similarityOVO[0]+=similarityOVO[i];

	cout<<strTrainingFileName<<endl;
	cout<<"errAB= "<<avgErrAB<<endl;
	cout<<"OVO: Pareto KL divergence similarity is avg= "<<avg<<endl;
	cout<<"OVO: Pareto KL divergence similarity is var= "<<var<<endl;

	checkRestCalsses(problem, v_v_Instance, v_nLabel, nrClass, similarityGamma,numFeature);
	cout<<"training size "<<v_nLabel.size()<<" numFeture "<<numFeature <<"gamma "<<similarityGamma<<endl;

}

