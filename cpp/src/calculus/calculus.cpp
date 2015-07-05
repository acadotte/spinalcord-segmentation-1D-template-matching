#define IND2(I, J, i, j) (j+i*J)

#include <cstring>
#include <iostream>
#include <math.h>
#include <numeric>
#include <vector>
#include "calculus.h"
#include "itkImage.h"
#include "itkImportImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkGradientRecursiveGaussianImageFilter.h"
#include "itkCovariantVector.h"
#include "itkImageRegionConstIterator.h"


using namespace std;

inline long int IND(int I, int J, int K, long int i, long int j, long int k) {
	return ((i) * J * K + j * K + k);
}

void alter_in_place(int A_i, int A_j, double* A) {
    if (A_i != 3)
        throw "Must have 3 rows!";
    for (int i = 0; i < A_i; i++)  
        for (int j = 0; j < A_j; j++)
            A[IND2(A_i, A_j, i, j)] *= 2;
}

double scalar_product(vector<double> a, vector<double> b)
{
    if( a.size() != b.size() ) // error check
    {
        puts( "Error a's size not equal to b's size" ) ;
	cout<<"A/B Size: "<<a.size()<<" / "<<b.size()<<endl;
        return -1 ;  // not defined
    }

    // compute
    double product = 0;
    for (int i = 0; i <= a.size()-1; i++)
       product += (a[i])*(b[i]); // += means add to product
    return product;
}

void gradient_calc(int S_i, float* S, int i, int j, int k, float* data, int r, int s,  int t, unsigned short* output_label){
	
	//cout<<"Starting Definitions..."<<endl;	
	//define the pixel and image types	
	typedef float PixelType;
	typedef itk::Image<PixelType, 3> ImageType;
	typedef itk::CovariantVector< double, 3 > GradientPixelType;
	typedef itk::Image< GradientPixelType, 3 > GradientImageType;
	typedef itk::GradientRecursiveGaussianImageFilter<ImageType, GradientImageType> GradientFilterType;
		
	//Initialize a new image which will read the input label	
	ImageType::Pointer image = ImageType::New();
	ImageType::IndexType start;
	start[0] = 0;
	start[1] = 0; 
	start[2] = 0;
	ImageType::SizeType size;
	size[0] = k;
	size[1] = j;
	size[2] = i;
	ImageType::SpacingType spacing;
	spacing[0] = S[0]; 
	spacing[1] = S[1]; 
	spacing[2] = S[2];
	//cout<<"X_space="<<spacing[0]<<", Y_space="<<spacing[1]<<", Z_space="<<spacing[2]<<endl;
	image->SetSpacing( spacing );
	ImageType::RegionType region;
	region.SetSize( size );
	region.SetIndex( start );
	image->SetRegions( region );
	image->Allocate();

	//Create an image iterator to copy the input label into the image file
	typedef itk::ImageRegionIterator< ImageType > IteratorType;
	IteratorType it( image, image->GetRequestedRegion() );

	int count = 0;	
	while(!it.IsAtEnd()){	
		it.Set( data[ count ] );
		++it;
		count++;
	}
	
	//Calculate the vector gradient of the image
	GradientFilterType::Pointer gradientMapFilter = GradientFilterType::New();
	gradientMapFilter->SetInput( image );
	gradientMapFilter->SetSigma( 1.0 );
	gradientMapFilter->Update();

	//Creates a new image and iterator of the gradient
	GradientImageType::Pointer image2 = gradientMapFilter->GetOutput();
	typedef itk::ImageRegionConstIterator< GradientImageType > IteratorType2;
	IteratorType2 it2( image2, image2->GetRequestedRegion() );
	//ImageType::IndexType idx = it2.GetIndex();

	//Outputs the input image to test if the input worked correctly
	ImageType::RegionType region2 = image2->GetLargestPossibleRegion();

	long int count2 = 0;	
	while(!it2.IsAtEnd()){
	//while(count2 < (r * s * t - 1)){		
		float magnitude = sqrt(it2.Get()[0] * it2.Get()[0] + it2.Get()[1] * it2.Get()[1] + it2.Get()[2] * it2.Get()[2]);
		output_label[ count2 ] = magnitude ;
		count2++;
		++it2;

	}
	
}

void convolve_arrays_fast(float matchThreshold, int offsetThreshold, int method, int testArray_i, int testArray_j, float* testArray, 
	int templateArray_i, int templateArray_j, float* templateArray, int templateEdgeIndex_i, int* templateEdgeIndex, int* matchPercent_i, int* matchPercent_j, float **matchPercent,
	int* templateIndex_i, int* templateIndex_j, int **templateIndex, int* edgeIndex_i, int* edgeIndex_j, int **edgeIndex, int* newEdge_i, int **newEdge) {


	(*matchPercent_i) = testArray_i;
	(*matchPercent_j) = templateArray_i;
	(*matchPercent) = new float[testArray_i * templateArray_i];

	(*templateIndex_i) = testArray_i;
	(*templateIndex_j) = templateArray_i;
	(*templateIndex) = new int[testArray_i * templateArray_i];   

	(*edgeIndex_i) = testArray_i;
	(*edgeIndex_j) = templateArray_i;
	(*edgeIndex) = new int[testArray_i * templateArray_i];     

	(*newEdge_i) = testArray_i;
	(*newEdge) = new int[testArray_i];

	for (int a = 0; a < (testArray_i * templateArray_i); a++) {   
		(*matchPercent)[a] = 0;
		(*templateIndex)[a] = 0;
		(*edgeIndex)[a] = 0;
	}

	int matchCount;
	int templateIndexCount;
	int offsetIndexCount;
	int bestIndex;
	int bestOffset;
	float bestScore;
	float temp_sum1;
	float temp_sum2;
	int count;
	float test_avg;
	float template_avg;
	
	for (int q = 0; q < testArray_i; q++) { //test each line of the test array
		if (testArray[q * testArray_j] == 0){ //checks to see if the first value in the array is 0 (cases in which only a small portion of the image is being tested), and bypasses that line in this case		
			continue;
		}		
		matchCount = 0;
		templateIndexCount = 0;
		offsetIndexCount = 0;
		bestIndex = 0;
		bestOffset = 0;

		for (int r = 0; r < templateArray_i; r++) { //test each line of the template against each test line

			bestScore = 0.0;
			//calculate the average value for the test array        
			for (int x = 0; x < min(testArray_j, offsetThreshold); x++) { //number of elements in the test image or the threshold of how many will be tested
				//cout<<"q = "<<q<<", r = "<<r<<", x = "<<x<<endl;		    
				temp_sum1 = 0.0;
				temp_sum2 = 0.0;
				count = 0;
		
				for (int z = 0; z < testArray_j; z++) {					
					if ((z - x) >= 0 && ((z - x) <= templateArray_j)) { 				
						temp_sum1 += testArray[q * testArray_j + z];
						temp_sum2 += templateArray[r * templateArray_j + (z-x)];
						count += 1;								
					}
				}

				test_avg = temp_sum1 / count;
				template_avg = temp_sum2 / count;
				
				
				float sum1 = 0.0;
				float sum2 = 0.0;
				float sum3 = 0.0;
				float sum4 = 0.0;
				//if (test_avg != 0){ //Checks for an empty array; used for testing purposes
				for (int s = 0; s < templateArray_j; s++) { //number of elements in each template line
					if (((x + s) < testArray_j) || ((x + s) < templateArray_j)) {
						sum1 += (templateArray[r * templateArray_j + s] - template_avg) * (testArray[q * testArray_j + (s + x)] - test_avg);
						sum2 += (templateArray[r * templateArray_j + s] - template_avg) * (templateArray[r * templateArray_j + s] - template_avg);
						sum3 += (testArray[q * testArray_j + (s + x)] - test_avg) * (testArray[q * testArray_j + (s + x)] - test_avg);
						sum4 += (templateArray[r * templateArray_j + s]) * (testArray[q * testArray_j + (s + x)]);
					}
					//cout<<"Sum1/2/3/4: "<<sum1<<"/"<<sum2<<"/"<<sum3<<"/"<<sum4<<endl;
				}
				//}
				float output;				
		
				if (sum1 != 0 && sum2 != 0) {
					output = sum1 / (sqrt(sum2) * sqrt(sum3));
				}
				else {	
					output = 0;
				}
				
				//cout<<"Output="<<output<<endl;
				//Calculate the % match

				if (output > bestScore){
					bestScore = output;
					bestIndex = r;
			    		bestOffset = x; //finds the element shift in the best template match
					//cout<<"bestIndex/BestOffset:"<<bestIndex<<"/"<<bestOffset<<endl;
				}
		    
		    	
			}

			if (bestScore > matchThreshold) { //Added the second condition in here to ensure that templates taken from the brain aren't applied to the spine. Need to think of a better way to make this more flexible though.
				if ((q / testArray_i) < 0.2 || ((q / testArray_i) > 0.2 && templateEdgeIndex[bestIndex] < 30)) {
					(*matchPercent)[q * templateArray_i + matchCount] = bestScore;
					matchCount += 1;
					(*templateIndex)[q * templateArray_i + templateIndexCount] = bestIndex; //finds the template line index which contains the best match
					templateIndexCount += 1;
					(*edgeIndex)[q * templateArray_i + offsetIndexCount] = bestOffset + templateEdgeIndex[bestIndex]; //finds the element shift in the best template match
					offsetIndexCount += 1;
				}	
			}
		}
	}
	
	for (int i = 0; i < testArray_i; i++) {
		count = 0;
		int sum = 0;
		for (int j = 0; j < templateArray_i; j++) {
			float testPoint = (*edgeIndex)[i * templateArray_i + j];
			if (testPoint != 0) {			
				sum += (*edgeIndex)[i * templateArray_i + j];
				count += 1;
			}
		}
		if (count >= 10) {
			(*newEdge)[i] = int(sum / count);
		}
		else{
			(*newEdge)[i] = 0;
		}
	}		
}


//This is the same method as above except the 'offset' variable has been removed, i.e., it will convolve a single instance of an array, but not try the offsets of that array.
void convolve_arrays_fast_2(float matchThreshold, int method, int testArray_i, int testArray_j, float* testArray, 
	int templateArray_i, int templateArray_j, float* templateArray, int templateEdgeIndex_i, int* templateEdgeIndex, int* matchPercent_i, int* matchPercent_j, float **matchPercent,
	int* templateIndex_i, int* templateIndex_j, int **templateIndex, int* edgeIndex_i, int* edgeIndex_j, int **edgeIndex, int* newEdge_i, int **newEdge) {
    
	(*matchPercent_i) = testArray_i;
	(*matchPercent_j) = templateArray_i;
	(*matchPercent) = new float[testArray_i * templateArray_i];

	(*templateIndex_i) = testArray_i;
	(*templateIndex_j) = templateArray_i;
	(*templateIndex) = new int[testArray_i * templateArray_i];   

	(*edgeIndex_i) = testArray_i;
	(*edgeIndex_j) = templateArray_i;
	(*edgeIndex) = new int[testArray_i * templateArray_i];     

	(*newEdge_i) = testArray_i;
	(*newEdge) = new int[testArray_i];

	for (int a = 0; a < (testArray_i * templateArray_i); a++) {   
		(*matchPercent)[a] = 0;
		(*templateIndex)[a] = 0;
		(*edgeIndex)[a] = 0;
	}

	int matchCount;
	int templateIndexCount;
	int offsetIndexCount;
	int bestIndex;
	float bestScore;
	float temp_sum1;
	float temp_sum2;
	int count;
	float test_avg;
	float template_avg;
	
	//cout<<"testArray_i="<<testArray_i<<" testArray_j="<<testArray_j<<"templateArray_i="<<templateArray_i<<"templateArray_j="<<templateArray_j<<endl;
	for (int q = 0; q < testArray_i; q++) { //test each line of the test array
		if (testArray[q * testArray_j] == 0){ //checks to see if the first value in the array is 0 (cases in which only a small portion of the image is being tested), and bypasses that line in this case		
			continue;
		}		
		matchCount = 0;
		templateIndexCount = 0;
		offsetIndexCount = 0;
		bestIndex = 0;

		//for (int r = 0; r < 1000; r++) { //test each line of the template against each test line
		for (int r = 0; r < templateArray_i; r++) { //test each line of the template against each test line

			bestScore = 0.0;
			//calculate the average value for the test array        
			



			//cout<<"q = "<<q<<", r = "<<r<<", x = "<<x<<endl;		    
			temp_sum1 = 0.0;
			temp_sum2 = 0.0;
			count = 0;
	
			for (int z = 0; z < testArray_j; z++) {					
				if (z >= 0 && (z <= templateArray_j)) { 				
					temp_sum1 += testArray[q * testArray_j + z];
					temp_sum2 += templateArray[r * templateArray_j + z];
					count += 1;								
				}
			}

			test_avg = temp_sum1 / count;
			template_avg = temp_sum2 / count;
			
			
			float sum1 = 0.0;
			float sum2 = 0.0;
			float sum3 = 0.0;
			float sum4 = 0.0;
			//if (test_avg != 0){ //Checks for an empty array; used for testing purposes
			for (int s = 0; s < templateArray_j; s++) { //number of elements in each template line
				if ((s < testArray_j) && (s < templateArray_j)) { //THIS USED TO BE AN OR, SWITCHED TO &&
					sum1 += (templateArray[r * templateArray_j + s] - template_avg) * (testArray[q * testArray_j + s] - test_avg);
					sum2 += (templateArray[r * templateArray_j + s] - template_avg) * (templateArray[r * templateArray_j + s] - template_avg);
					sum3 += (testArray[q * testArray_j + s] - test_avg) * (testArray[q * testArray_j + s] - test_avg);
					sum4 += (templateArray[r * templateArray_j + s]) * (testArray[q * testArray_j + s]);
				}
				//cout<<"Sum1/2/3/4: "<<sum1<<"/"<<sum2<<"/"<<sum3<<"/"<<sum4<<endl;
			}
			//}
			float output;				
	
			if (sum1 != 0 && sum2 != 0) {
				output = sum1 / (sqrt(sum2) * sqrt(sum3));
			}
			else {	
				output = 0;
			}
			
			//cout<<"Output="<<output<<endl;
			//Calculate the % match

			if (output > bestScore){
				bestScore = output;
				bestIndex = r;
			}
		    

			if (bestScore > matchThreshold) { //Added the second condition in here to ensure that templates taken from the brain aren't applied to the spine. Need to think of a better way to make this more flexible though.
				if ((q / testArray_i) < 0.2 || ((q / testArray_i) > 0.2 && templateEdgeIndex[bestIndex] < 30)) {
					(*matchPercent)[q * templateArray_i + matchCount] = bestScore;
					matchCount += 1;
					(*templateIndex)[q * templateArray_i + templateIndexCount] = bestIndex; //finds the template line index which contains the best match
					templateIndexCount += 1;
					(*edgeIndex)[q * templateArray_i + offsetIndexCount] = templateEdgeIndex[bestIndex]; //finds the element shift in the best template match
					offsetIndexCount += 1;
				}	
			}
		}
	}
	
	for (int i = 0; i < testArray_i; i++) {
		count = 0;
		int sum = 0;
		int min_val = 1000;
		int min_index = 0;
		for (int j = 0; j < templateArray_i; j++) {
			float testPoint = (*edgeIndex)[i * templateArray_i + j];
			if (testPoint != 0) {			
				sum += (*edgeIndex)[i * templateArray_i + j];
				count += 1;
			}
			if (testPoint != 0 && testPoint < min_val) {
				min_val = testPoint;
				min_index = i * templateArray_i + j;
			}

		}
		
		if (count >= 10 && method == 1) {
			//(*newEdge)[i] = int(sum / count);
			(*newEdge)[i] = (*edgeIndex)[min_index];
		}
		//else if (count >= 10 && method == 2) { // Need to have a method for calculating the median here
		//	(*newEdge)[i] = int(sum / count);
		//}
		else{
			(*newEdge)[i] = 0;
		}
	}		
}


//A recursive method to find the edges of the spine
void convolve_arrays_recursive(float matchThreshold, float matchThresholdOriginal, int minimumMatches, int method, int loopCounter, int lastEdgeIndex, int testArray_i, int testArray_j, float* testArray,
	int templateArray_i, int templateArray_j, float* templateArray, int templateEdgeIndex_i, int* templateEdgeIndex, int edgeIndexOut_i, int edgeIndexOut_j, int* edgeIndexOut, 
	int matchOut_i, int matchOut_j, float* matchOut, int matched_template_out_i, int matched_template_out_j, int* matched_template_out, int matchPercentOut_i, int matchPercentOut_j, int* matchPercentOut) {

	int bestIndex;
	float bestScore;
	float temp_sum1;
	float temp_sum2;
	int count;
	float test_avg;
	float template_avg;
	int edgeSearchThreshold = 7;
	//int minimumMatches = 50;
	float thresholdIncr = 0.02;
	int lastEdgeIndexLocal = lastEdgeIndex;
	int loopCounterLocal = loopCounter;
	float thresholdLocal = matchThreshold;
	int* edgeIndex = new int[templateArray_i];
	//int array_shown = 0;
	
	//Initializes the arrays at 0
	for (int a = 0; a < (templateArray_i); a++) {   
		edgeIndex[a] = 0;
	}
	
	int num_edges_to_record = matched_template_out_j;
	int matched_array_counter = 0;
	//int matchPercentCounter = 0;

	bestIndex = 0;

	for (int r = 0; r < templateArray_i; r++) { //test each line of the template against each test line
		if ((templateEdgeIndex[r] > (lastEdgeIndexLocal + edgeSearchThreshold) || templateEdgeIndex[r] < (lastEdgeIndexLocal - edgeSearchThreshold)) && lastEdgeIndex != 0) { 
			//skips the specific template array if its edgeIndex is out of the desired search bounds
			//cout<<"Skipping r="<<r<<"; template edgeIndex="<<templateEdgeIndex[r]<<endl;
			continue;
		}
	
		bestScore = 0.0;
		//calculate the average value for the test array        

		temp_sum1 = 0.0;
		temp_sum2 = 0.0;
		count = 0;

		for (int z = 0; z < (templateEdgeIndex[r] + edgeSearchThreshold); z++) {					
			if (z >= 0 && (z <= templateArray_j)) { 				
				temp_sum1 += testArray[loopCounter * testArray_j + z];
				temp_sum2 += templateArray[r * templateArray_j + z];
				count += 1;								
			}
		}

		test_avg = temp_sum1 / count;
		template_avg = temp_sum2 / count;
	
		float sum1 = 0.0;
		float sum2 = 0.0;
		float sum3 = 0.0;
		float sum4 = 0.0;

		for (int s = 0; s < (templateEdgeIndex[r] + edgeSearchThreshold); s++) { //number of elements in each template line
			if ((s < testArray_j) && (s < templateArray_j)) { //THIS USED TO BE AN OR, SWITCHED TO &&
				sum1 += (templateArray[r * templateArray_j + s] - template_avg) * (testArray[loopCounter * testArray_j + s] - test_avg);
				sum2 += (templateArray[r * templateArray_j + s] - template_avg) * (templateArray[r * templateArray_j + s] - template_avg);
				sum3 += (testArray[loopCounter * testArray_j + s] - test_avg) * (testArray[loopCounter * testArray_j + s] - test_avg);
				sum4 += (templateArray[r * templateArray_j + s]) * (testArray[loopCounter * testArray_j + s]);
			}
			//cout<<"Sum1/2/3/4: "<<sum1<<"/"<<sum2<<"/"<<sum3<<"/"<<sum4<<endl;
		}

		float output;				

		if (sum1 != 0 && sum2 != 0) {
			output = sum1 / (sqrt(sum2) * sqrt(sum3));
		}
		else {	
			output = 0;
		}
	
		//cout<<"Output="<<output<<endl;
		//Calculate the % match

		if (output > bestScore){
			bestScore = output;
			bestIndex = r;
		}
	    
		if (output > matchThreshold) {  
			edgeIndex[r] = templateEdgeIndex[bestIndex]; //finds the element shift in the best template match
			matched_template_out[loopCounter * num_edges_to_record + matched_array_counter] = r;
		
			if (matched_array_counter < num_edges_to_record) {
				matched_array_counter += 1;
			}
		}
	}

	count = 0;
	int sum = 0;
	int min_val = 10000;
	int max_val = 0;
	//int max_val = 0;
	int min_index = 0;
	int max_index = 0;
	for (int j = 0; j < templateArray_i; j++) {
		int testPoint = edgeIndex[j];
		if (testPoint != 0) {			
			sum += edgeIndex[j];
			count += 1;
		}
		if (testPoint != 0 && testPoint < min_val) {
			min_val = testPoint;
			min_index = j;
		}
		if (testPoint != 0 && testPoint > max_val) {
			max_val = testPoint;
			max_index = j;
		}
	}
	
	
	
	if (count >= minimumMatches && method == 1) { //1 = minimum index value
		matchPercentOut[loopCounter * matchPercentOut_j + 0] = count;
		matchPercentOut[loopCounter * matchPercentOut_j + 1] = thresholdLocal * 100;
		edgeIndexOut[loopCounter] = edgeIndex[min_index];
		matchOut[loopCounter] = thresholdLocal;
		loopCounterLocal += 1;
		thresholdLocal = matchThresholdOriginal;
		lastEdgeIndexLocal = edgeIndex[min_index];
		//matchPercentCounter = 0;

	}
	else if (count >= minimumMatches && method == 2) { // 2 = average index value
		matchPercentOut[loopCounter * matchPercentOut_j + 0] = count;
		matchPercentOut[loopCounter * matchPercentOut_j + 1] = thresholdLocal * 100;
		//edgeIndexOut[loopCounter] = int(sum / count) + 2; //adds a constant based on testing to get closer to the edge. 2 works good for high res images
		//edgeIndexOut[loopCounter] = int(sum / count) - 1; //adds a constant based on testing to get closer to the edge. Closer to 0 is better for low res images
		edgeIndexOut[loopCounter] = int(sum / count) + 1; //adds a constant based on testing to get closer to the edge. Closer to 0 is better for low res images
		matchOut[loopCounter] = thresholdLocal;
		loopCounterLocal += 1;
		thresholdLocal = matchThresholdOriginal;
		lastEdgeIndexLocal = int(sum / count);
		//matchPercentCounter = 0;

	}
	else if (count >= minimumMatches && method == 3) { // 3 = maximum index value
		matchPercentOut[loopCounter * matchPercentOut_j + 0] = count;
		matchPercentOut[loopCounter * matchPercentOut_j + 1] = thresholdLocal * 100;
		edgeIndexOut[loopCounter] = edgeIndex[max_index];
		matchOut[loopCounter] = thresholdLocal;
		loopCounterLocal += 1;
		thresholdLocal = matchThresholdOriginal;
		lastEdgeIndexLocal = edgeIndex[max_index];
		//matchPercentCounter = 0;

	}
	else if (count >= minimumMatches && method == 4) { // 4 = average of method 2 and method 4
		matchPercentOut[loopCounter * matchPercentOut_j + 0] = count;
		matchPercentOut[loopCounter * matchPercentOut_j + 1] = thresholdLocal * 100;
		edgeIndexOut[loopCounter] = int((edgeIndex[max_index] + sum / count) / 2);//int((edgeIndex[min_index] + sum / count) / 2);
		matchOut[loopCounter] = thresholdLocal;
		loopCounterLocal += 1;
		thresholdLocal = matchThresholdOriginal;
		lastEdgeIndexLocal = edgeIndex[min_index] + 1; //int((edgeIndex[min_index] + sum / count) / 2);
		//matchPercentCounter = 0;

	}
	else if (count >= minimumMatches && method == 5) { // 4 = average of method 1 and method 2
		matchPercentOut[loopCounter * matchPercentOut_j + 0] = count;
		matchPercentOut[loopCounter * matchPercentOut_j + 1] = thresholdLocal * 100;
		edgeIndexOut[loopCounter] = int((edgeIndex[min_index] + sum / count) / 2);//int((edgeIndex[min_index] + sum / count) / 2);
		matchOut[loopCounter] = thresholdLocal;
		loopCounterLocal += 1;
		thresholdLocal = matchThresholdOriginal;
		lastEdgeIndexLocal = edgeIndex[min_index] + 1; //int((edgeIndex[min_index] + sum / count) / 2);
		//matchPercentCounter = 0;

	}
	else if (thresholdLocal < 0.90){ //for cases where an edge can't be found
		matchPercentOut[loopCounter * matchPercentOut_j + 0] = 0;
		matchPercentOut[loopCounter * matchPercentOut_j + 1] = 0;
		edgeIndexOut[loopCounter] = 0;
		matchOut[loopCounter] = 0.0;
		loopCounterLocal += 1;
		thresholdLocal = matchThresholdOriginal;
		lastEdgeIndexLocal = 0; //Setting this to zero will allow the algo to search all templates for the next array, when one wasn't found for the prior array
		//matchPercentCounter = 0;

	}
	else{
		//When there is no index found, a recursion is called with the same loopCounter (i.e., same testArray line being tested), a lower threshold			
		loopCounterLocal += 0;
		thresholdLocal = matchThreshold - thresholdIncr;
		//cout<<"No match found, new threshold="<<thresholdLocal<<endl;						
	}
	delete[] edgeIndex; //deallocates memory for edgeIndex since it will be recreated in each recurssion
	//cout<<"Pre-Recursion LoopCounter / threshold = "<<loopCounterLocal<<" / "<<thresholdLocal<<endl;	
			
	if(loopCounterLocal < (testArray_i - 1)) {	
		convolve_arrays_recursive(thresholdLocal, matchThresholdOriginal, minimumMatches, method, loopCounterLocal, lastEdgeIndexLocal, testArray_i, testArray_j, testArray, templateArray_i, templateArray_j, templateArray, templateEdgeIndex_i, templateEdgeIndex, edgeIndexOut_i, edgeIndexOut_j, edgeIndexOut, matchOut_i, matchOut_j, matchOut, matched_template_out_i, matched_template_out_j, matched_template_out, matchPercentOut_i, matchPercentOut_j, matchPercentOut);
	}
}


void smoothing_filter_1D(int axial_smoothing, int method, int kernel_size, int input_array_i, int input_array_j, float* input_array, int* output_array_i, int* output_array_j, float **output_array) {
//Methods: 1 = median, 2 = mean

	(*output_array_i) = input_array_i;
	(*output_array_j) = input_array_j;
	(*output_array) = new float[input_array_i * input_array_j];

	float* temp_array = new float[kernel_size];
	int zero_counter = 0;
	int orig_kernel_size = kernel_size;

	float threshold_p = 0.25;

	for (int a = 0; a < kernel_size; a++) {
		temp_array[a] = 0;
	}

	if (axial_smoothing == 1) {
		for (int n = 0; n < (input_array_i * input_array_j); n++) {
			zero_counter = 0;
			if (n > (input_array_i * input_array_j) * threshold_p) {
				kernel_size = orig_kernel_size * 2 + 1;
			}
			else{
				kernel_size = orig_kernel_size;
			}			
			
			for (int z = 0; z < kernel_size; z++) {
				if (n > ((input_array_i * input_array_j) - int(kernel_size / 2))) { //The end of the array. The result is that the last kernel size elements will all have the same value
					temp_array[z] = input_array[input_array_i * input_array_j - z];
					if (temp_array[z] == 0) {zero_counter += 1;}
				}
				else if (n > int(kernel_size / 2)) {				
					temp_array[z] = input_array[n - int(kernel_size / 2) + z];
					if (temp_array[z] == 0) {zero_counter += 1;}
				}
				else { //when n is less than the kernel size, i.e., at the beginning of the array. The result is that the beging kernel_size values will all be the same.
					temp_array[z] = input_array[z];
					if (temp_array[z] == 0) {zero_counter += 1;}
				}
			}			

			//cout<<"Kernel Size: "<<kernel_size<<"; Zero Element Size: "<<zero_counter<<endl;
			if (kernel_size > zero_counter) {
				if (method == 1) {				
					(*output_array)[n] = GetMedian(temp_array, kernel_size, kernel_size - zero_counter);
				}
				else if (method == 2) {
					(*output_array)[n] = GetMean(temp_array, kernel_size);
				}
			}
			else {
				(*output_array)[n] = 0;
			}
		}
	}	
	else { //non-axial smoothing, i.e., longitudinal smoothing
		for (int n = 0; n < (input_array_j); n++) {
			for (int m = 0; m < (input_array_i); m ++) {
				zero_counter = 0;
				if (n > (input_array_i * input_array_j) * threshold_p) {
					kernel_size = orig_kernel_size * 2 + 1;
				}
				else{
					kernel_size = orig_kernel_size;
				}	
				for (int z = 0; z < kernel_size; z++) {
					if (m > (input_array_i - int(kernel_size / 2))) {
						temp_array[z] = input_array[(input_array_i - z) * input_array_j + n];
						if (temp_array[z] == 0) {zero_counter += 1;}
					}
					else if (m > int(kernel_size / 2)) {
						temp_array[z] = input_array[(m - int(kernel_size / 2) + z) * input_array_j + n];
						if (temp_array[z] == 0) {zero_counter += 1;}
					}
					else { //at the beginning of the array
						temp_array[z] = input_array[z * input_array_j + n];
						if (temp_array[z] == 0) {zero_counter += 1;}
					}
				}
				//cout<<"Kernel Size: "<<kernel_size<<"; Zero Element Size: "<<zero_counter<<endl;
				if (kernel_size > zero_counter) {
					if (method == 1) {
						(*output_array)[m * input_array_j + n] = GetMedian(temp_array, kernel_size, kernel_size - zero_counter);
					}
					else if (method == 2) {
						(*output_array)[m * input_array_j + n] = GetMean(temp_array, kernel_size);
					}
				}
				else {
					(*output_array)[m * input_array_j + n] = 0;
				}
			}
		}
	}
}

float GetMedian(float daArray[], int arraySize, int iSize) {
    //ArraySize is the total length of the input array, iSize is the length of the non-zero elements
    // Allocate an array of the same size and sort it.
    float* dpSorted = new float[iSize];
    int count = 0;

    for (int i = 0; i < arraySize; ++i) {
	if (daArray[i] > 0) {        
    	    dpSorted[count] = daArray[i];
	    count += 1;
	}
    }

    for (int i = iSize - 1; i > 0; --i) {
        for (int j = 0; j < i; ++j) {
            if (dpSorted[j] > dpSorted[j+1]) {
                double dTemp = dpSorted[j];
                dpSorted[j] = dpSorted[j+1];
                dpSorted[j+1] = dTemp;
            }
        }
    }

    // Middle or average of middle values in the sorted array.
    float dMedian = 0.0;
    if ((iSize % 2) == 0) {
        dMedian = (dpSorted[iSize/2] + dpSorted[(iSize/2) - 1])/2.0;
    } else {
        dMedian = dpSorted[iSize/2];
    }
    delete [] dpSorted;
    return dMedian;
}


float GetMean(float daArray[], int iSize) {
    //ArraySize is the total length of the input array, iSize is the length of the non-zero elements
    float dSum = daArray[0];
    int count = 0;

    for (int i = 1; i < iSize; ++i) {
	if (daArray[i] > 0) {
            dSum += daArray[i];
	    count += 1;
	}
    }
    return dSum/count;
}



//Jaccard Index: calculates the intersection / union of two images to find their similarity
float jaccard_index(int a, int b, int c, unsigned short* label_1, int d, int e, int f, unsigned short* label_2) {
	
	int unionCalc = 0;
	int intersectCalc = 0;
	float jaccard = 0.0;

	cout<<"Label 1 Shape: "<<a<<"/"<<b<<"/"<<c<<endl;
	cout<<"Label 2 Shape: "<<d<<"/"<<e<<"/"<<f<<endl;

	for (int x = 0; x < a; x++){
		for (int y = 0; y < b; y++){
			for (int z = 0; z < c; z++) {
				if (label_1[IND(a, b, c, x, y, z)] == 1 || label_2[IND(d, e, f, x, y, z)] == 1) {
					//cout<<"UNION MATCH: "<<unionCalc<<endl;					
					unionCalc += 1;
				}
				if (label_1[IND(a, b, c, x, y, z)] == 1 && label_2[IND(d, e, f, x, y, z)] == 1) {
					//cout<<"INTERSECT MATCH"<<intersectCalc<<endl;
					intersectCalc += 1;
				}
			}
		}
	}
	//cout<<"INTERSECT: "<<intersectCalc<<", UNION: "<<unionCalc<<endl;
	
	jaccard = float(intersectCalc) / float(unionCalc) * 100;
	
	return jaccard;
}

//Dice Coefficient: calculates the intersection / union of two images to find their similarity
float dice_coeff(int a, int b, int c, unsigned short* label_1, int d, int e, int f, unsigned short* label_2, int r, int s,  int t, unsigned short* output_label) {
	
	int label_1_count = 0;
	int label_2_count = 0;
	int intersectCalc = 0;
	float dice = 0.0;
	int lab1_val = 0;
	int lab2_val = 0;

	cout<<"Label 1 Shape: "<<a<<"/"<<b<<"/"<<c<<endl;
	cout<<"Label 2 Shape: "<<d<<"/"<<e<<"/"<<f<<endl;

	for (int x = 0; x < a; x++){
		for (int y = 0; y < b; y++){
			for (int z = 0; z < c; z++) {
				lab1_val = label_1[IND(a, b, c, x, y, z)];
				lab2_val = label_2[IND(d,e,f, x, y, z)];
				if (lab1_val == 1) {
					label_1_count += 1;
				}
				if (lab2_val == 1) {
					label_2_count += 1;
				}
				if (lab1_val == 1 && lab2_val == 1) {
					//cout<<"INTERSECT MATCH"<<intersectCalc<<endl;
					intersectCalc += 1;
				}
				else if(lab1_val == 1 && lab2_val == 0) {
					output_label[IND(r,s,t, x, y, z)] = 2;
					//cout<<"Found point @"<<x<<"/"<<y<<"/"<<z<<endl;
				}
				else if(lab1_val == 0 && lab2_val == 1) {
					output_label[IND(r,s,t, x, y, z)] = 3;
					//cout<<"Found point @"<<x<<"/"<<y<<"/"<<z<<endl;
				}
			}
		}
	}
	//cout<<"INTERSECT: "<<intersectCalc<<", UNION: "<<unionCalc<<endl;
	
	dice = 2 * float(intersectCalc) / (float(label_1_count) + float(label_2_count));
	
	return dice;
}

float area_difference_output(int targ_y_idx, int a, int b, int c, unsigned short* label_1, int d, int e, int f, unsigned short* label_2, int r, int s,  int t, unsigned short* output_label) {
	//Measures the oversegmentation of one vs the other. Reports whichever is more oversegmented
	int label_1_count = 0;
	int label_2_count = 0;
	float area_diff = 0.0;
	int lab1_val = 0;
	int lab2_val = 0;

	//cout<<"Label 1 Shape: "<<a<<"/"<<b<<"/"<<c<<endl;
	//cout<<"Label 2 Shape: "<<d<<"/"<<e<<"/"<<f<<endl;

	for (int x = 0; x < a; x++){
		for (int y = 0; y < b; y++){
			for (int z = 0; z < c; z++) {
				lab1_val = label_1[IND(a, b, c, x, y, z)];
				lab2_val = label_2[IND(d,e,f, x, y, z)];
				if (lab1_val == 1 && y == targ_y_idx) {
					label_1_count += 1;
				}
				if (lab2_val == 1 && y == targ_y_idx) {
					label_2_count += 1;
				}
				
			}
		}
	}
	//cout<<"INTERSECT: "<<intersectCalc<<", UNION: "<<unionCalc<<endl;
	
	

	for (int x = 0; x < a; x++){
		for (int y = 0; y < b; y++){
			for (int z = 0; z < c; z++) {
				lab1_val = label_1[IND(a, b, c, x, y, z)];
				lab2_val = label_2[IND(d,e,f, x, y, z)];
				
				if(lab1_val == 1 && lab2_val == 0 && label_1_count > label_2_count) {
					output_label[IND(r,s,t, x, y, z)] = 3;
					//cout<<"Found point @"<<x<<"/"<<y<<"/"<<z<<endl;
				}
				else if(lab1_val == 0 && lab2_val == 1 && label_2_count > label_1_count) {
					output_label[IND(r,s,t, x, y, z)] = 3;
					//cout<<"Found point @"<<x<<"/"<<y<<"/"<<z<<endl;
				}
			}
		}
	}



	if(label_1_count > label_2_count){
		area_diff = (label_1_count - label_2_count);
	}
	else{
		area_diff = (label_2_count - label_1_count);
	}
	cout<<"Voxels - Label_1: "<<label_1_count<<", Label_2: "<<label_2_count<<endl;
	
	return area_diff;
}

//find the nearest surface to a given edge
float find_nearest_surface(int offset, int min_y, int max_y, int S_i, float* S, int a, int b, int c, int d, int e, int f, unsigned short* label_2){
	
	int start_x = max(0, a - offset);
	//int start_y = max(0, b - offset);
	int start_z = max(0, c - offset);
	int stop_x = min(d, a + offset);
	//int stop_y = min(e, b + offset);
	int stop_z = min(f, c + offset);
	int curr_label = 0;
	int last_label = 0;
	float distance = 0;
	float min_dist = 10000.0;

	//NEED TO MAKE SURE THIS ORDER IS CORRECT
	float x_size = S[0];
	float y_size = S[1];
	float z_size = S[2];

	int near_x = 0;
	int near_y = 0;
	int near_z = 0;


	curr_label = 0;			
	
	//cout<<"Start Point: "<<start_x<<"/"<<start_y<<"/"<<start_z<<endl;
	//cout<<"Stop Point: "<<stop_x<<"/"<<stop_y<<"/"<<stop_z<<endl;
	for(int y = min_y; y < max_y; y++){
		for(int x = start_x; x < stop_x; x++){
			for(int z = start_z; z < stop_z; z++){
				curr_label = label_2[IND(d, e, f, x, y, z)];
				if ((curr_label != 0 && last_label == 0) || (curr_label == 0 && last_label != 0)) {
					//found an edge
					float d = (abs((x - a)) * x_size) * (abs((x - a)) * x_size) + (abs((y - b)) * y_size) * (abs((y - b)) * y_size)  + (abs((z - c)) * z_size) * (abs((z - c)) * z_size);
					distance = sqrt(d);
					if(distance < min_dist){
						min_dist = distance;
						near_x = x;
						near_y = y;
						near_z = z;
					}	
				}
				last_label = curr_label;
			}
		}
	}
	//cout<<"Original Point: "<<a<<"/"<<b<<"/"<<c<<"; Nearest surface edge at: "<<near_x<<"/"<<near_y<<"/"<<near_z<<"; Min Dist: "<<min_dist<<endl;
	return min_dist;
}

//Hausdorff Distance: the maximum of the minimum distances between two surfaces
float haus_dist(int offset, int average_out, int S_i, float* S, int a, int b, int c, unsigned short* label_1, int d, int e, int f, unsigned short* label_2) {
	
	int last_label = 0;
	float max_dist = 0;
	int curr_label = 0;

	float dist_sum = 0;
	int dist_count = 0;

	int max_x = 0;
	int max_y = 0;
	int max_z = 0;

	int min_b = 0;
	int max_b = 500;

	int y1_min_set = 0;
	int y2_min_set = 0;

	int found_value_on_y_plane_1 = 0;
	int found_value_on_y_plane_2 = 0;

	cout<<"Calculating Hausdorff distance..."<<endl;
	//cout<<"Label 1 Shape: "<<a<<"/"<<b<<"/"<<c<<endl;
	//cout<<"Label 2 Shape: "<<d<<"/"<<e<<"/"<<f<<endl;

	//Check for the "longest" spine and set that as the furtherest search area
	for (int y = 0; y < b; y++){
		found_value_on_y_plane_1 = 0;
		found_value_on_y_plane_2 = 0;
		for (int x = 0; x < a; x++){
			for (int z = 0; z < c; z++) {
				if(x > 0 && y > 0 && z > 0){
					//whichever mask has a starting y value which is higher will get set as the minimum
					curr_label = label_1[IND(a, b, c, x, y, z)];
					if(y1_min_set == 0 && curr_label == 1){
						y1_min_set = 1;
						min_b = y;
					}
					if(curr_label == 1){
						found_value_on_y_plane_1 = 1;
					}

					curr_label = label_2[IND(d, e, f, x, y, z)];
					if(y2_min_set == 0 && curr_label == 1){
						y2_min_set = 1;
						min_b = y;
					}
					if(curr_label == 1){
						found_value_on_y_plane_2 = 1;
					}
				}
			}	
		}
		if(y > min_b && (found_value_on_y_plane_1 == 0 || found_value_on_y_plane_2 == 0) && max_b == 0){ //only sets the max if it hasn't been set yet 
			max_b = y;
		}
	}

					
					
						
					

	for (int y = min_b; y < max_b; y++){
		for (int x = 0; x < a; x++){
			for (int z = 0; z < c; z++) {
				if(x > 0 && y > 0 && z > 0){
					curr_label = label_1[IND(a, b, c, x, y, z)];
					if ((curr_label != 0 && last_label == 0) || (curr_label == 0 && last_label != 0)) {
						//found an edge
						//cout<<"Found edge at: "<<x<<"/"<<y<<"/"<<z<<endl;
						float temp_min_dist = find_nearest_surface(offset, min_b, max_b, S_i, S, x, y, z, d, e, f, label_2);
						dist_sum += temp_min_dist;
						dist_count += 1;
						//cout<<"Min Dist: "<<temp_min_dist<<endl;
						if(temp_min_dist > max_dist){
							max_dist = temp_min_dist;
							max_x = x;
							max_y = y;
							max_z = z;
							//cout<<"New Max Distance: "<<max_dist<<"; @ "<<max_x<<"/"<<max_y<<"/"<<max_z<<endl;
						}
					}
					last_label = curr_label;
				}
			}
		}
	}
	float avg = dist_sum / (1.0 * dist_count);
	
	if(average_out == 1){
		return avg;
	}
	else{
		return max_dist;
	}
}


//calculate the center of a mask in the axial plane
void centroid_calc(int a, int b, int c, unsigned short* label_1, int r, int s,  int t, unsigned short* output_label, int* pts_i, int* pts_j, float **pts) {

	float temp_x, temp_z;
	int count = 0;
	(*pts_i) = s;
	(*pts_j) = 3;
	(*pts) = new float[s * 3];
	int curr_label = 0;
	int num_centers = 1;

	for (int y = 0; y < b; y++){
		temp_x = 0;
		temp_z = 0;
		count = 0;
		for (int x = 0; x < a; x++){
			for (int z = 0; z < c; z++) {
				curr_label = label_1[IND(a, b, c, x, y, z)];
				if(curr_label == 1){
					temp_x += x;
					temp_z += z;
					count += 1;
				}
			}
		}
		
		if(count > 0){
			output_label[IND(r, s, t, int(temp_x / count), y, int(temp_z / count))] = 2;
			(*pts)[num_centers * 3] = int(temp_x / count);
			(*pts)[num_centers * 3 + 1] = y;
			(*pts)[num_centers * 3 + 2] = int(temp_z / count);
			num_centers += 1;
		}
	}
	(*pts)[0] = num_centers;
	(*pts)[1] = 0;
	(*pts)[2] = 0;
}

//count the number of filled voxels in a mask
int get_filled_voxels(int a, int b, int c, unsigned short* label_1) {

	int count = 0;

	for (int y = 0; y < b; y++){
		for (int x = 0; x < a; x++){
			for (int z = 0; z < c; z++) {
				if(label_1[IND(a, b, c, x, y, z)] > 0){
					count += 1;
				}
			}
		}
	}
	return count;
}


void create_submask_between_planes(int a, int b, int c, unsigned short* label_1, int r, int s,  int t, unsigned short* output_label, int inpts_i, int inpts_j, float *inpts) {
	//fills in the voxels on the output label only if the points are within the two planes given as inputs 
	//inpts is structured as an x by 3 array with this order: pt on plane 1, normal of plane 1, point on plane 2, normal of plane 2
	double dot_prod_1 = 0;
	double dot_prod_2 = 0;
	float temp[3] = {0, 0, 0};
	vector<double> e(3);
	vector<double> f1(3);
	f1[0] = (static_cast<double>(inpts[3]));
	f1[1] = (static_cast<double>(inpts[4]));
	f1[2] = (static_cast<double>(inpts[5]));
	vector<double> f2(3);
	f2[0] = (static_cast<double>(inpts[9]));
	f2[1] = (static_cast<double>(inpts[10]));
	f2[2] = (static_cast<double>(inpts[11]));

	double N = 0;

	for (int x = 0; x < a; x++){
		for (int y = 0; y < b; y++){
			for (int z = 0; z < c; z++) {
				//calculate the dot product of the first vector and the normalized vector from the test point to a point on the plane 1
			    	temp[0] = x - inpts[0];
				temp[1] = y - inpts[1];
				temp[2] = z - inpts[2]; //test point less pt on plane 1
				N = sqrt(temp[0] * temp[0] + temp[1] * temp[1] + temp[2] * temp[2]);

				e[0] = (static_cast<double>(temp[0]) / N);
				e[1] = (static_cast<double>(temp[1]) / N);
				e[2] = (static_cast<double>(temp[2]) / N);

				dot_prod_1 = scalar_product(e, f1);

				//calculate the dot product of the second vector and the normalized vector from the test point to a point on the plane 2
			    	temp[0] = x - inpts[6];
				temp[1] = y - inpts[7];
				temp[2] = z - inpts[8]; //test point less pt on plane 1


				N = sqrt(temp[0] * temp[0] + temp[1] * temp[1] + temp[2] * temp[2]);
				e[0] = (static_cast<double>(temp[0]) / N);	
				e[1] = (static_cast<double>(temp[1]) / N);
				e[2] = (static_cast<double>(temp[2]) / N);

				dot_prod_2 = scalar_product(e, f2);
				
				if (dot_prod_1 > 0 && dot_prod_2 < 0){
					output_label[IND(r,s,t, x, y, z)] = label_1[IND(a, b, c, x, y, z)];
					
				}
			}
		}
	}
}
