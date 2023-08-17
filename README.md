# Deep Learning Challenge

## Overview of the analysis:

The purpose of this analysis was to provide the nonprofit organization Alphabet Soup with a tool to help select the applicants for funding with the best chance of success. 

The data included 34,000 organizations that have received funding from Alphabet Soup in the past. The dataset information is as follows:
  *	EIN and NAME—Identification columns
  *	APPLICATION_TYPE—Alphabet Soup application type
  *	AFFILIATION—Affiliated sector of industry
  *	CLASSIFICATION—Government organization classification
  *	USE_CASE—Use case for funding
  *	ORGANIZATION—Organization type
  *	STATUS—Active status
  *	INCOME_AMT—Income classification
  *	SPECIAL_CONSIDERATIONS—Special considerations for application
  *	ASK_AMT—Funding amount requested
  *	IS_SUCCESSFUL—Was the money used effectively

As a part of this analysis, the data was processed, split with sklearn module’s train_test_split, then defined with tensorflow.models.Sequential before trained and tested for accuracy. The data was tested in a total of four separate ways. The first is within the Starter_Code file and then three optimization attempts within the AlphabetSoupCharity_Optimization file. The results were all exported into h5 files. 

## Results: 

  *	Data Preprocessing
    *	The target variable was the column ‘IS_SUCCESSFUL’
    *	The feature variables within the original model and optimization attempt 1 and 2 were the columns remaining after ‘IS_SUCCESSFUL’, ‘EIN’, and ‘NAME’ were removed.
    *	The feature variables within optimization attempt 3 were the columns remaining after ‘IS_SUCCESSFUL’, 'SPECIAL_CONSIDERATIONS', 'ORGANIZATION', 'STATUS', ‘EIN’, and ‘NAME’ were removed.
    *	‘EIN’ and ‘NAME’ data were removed because they are neither targets nor features.
    *	Data binning was implemented for the ‘APPLICATION_TYPE’ and ‘CLASSIFICATION’ categories in Original and Optimization Attempt 3.
    *	Data binning was implemented for the ‘APPLICATION_TYPE’, ‘CLASSIFICATION’, and ‘ASK_AMT’ categories in Optimization Attempt 1 and Optimization Attempt 2.

  * Compiling, Training, and Evaluating the Model
    * Original
      * Three layers:
        * “relu”, 80 neurons
        * “relu”, 30 neurons
        * “sigmoid”, 1
      * The model did not get to the target performance of 75% or higher.
      * Accuracy: 73.05%
    * Optimization Attempt 1
      * Three layers:
        * “relu”, 80 neurons
        * “relu”, 30 neurons
        * “sigmoid”, 1
      * The difference from the original was the ‘ASK_AMT’ binned into 2 types instead of 8747 unique values.
      * The model did not reach the target performance of 75% or higher.
      * Accuracy: 72.82%
    * Optimization Attempt 2
      * Three layers:
        * “tanh”, 80 neurons
        * “relu”, 30 neurons
        * “sigmoid”, 1
      * The differences from the original were the ‘ASK_AMT’ binned into 2 types instead of 8747 unique values AND the initial activation function was changed to tanh.
      * The model did not reach the target performance of 75% or higher.
      * Accuracy: 73.15%
    * Optimization Attempt 3
      * Three layers:
        * “tanh”, 80 neurons
        * “relu”, 30 neurons
        * “sigmoid”, 1
      * The differences from the original were the ‘ASK_AMT’ binned into 2 types instead of 8747 unique values, the initial activation function was changed to tanh, AND the features were lowered to 6 from 9.
      * The model did not reach the target performance of 75% or higher.
      * Accuracy: 72.68%
        
## Summary: 

Overall, the adjustments made to the models only affected the accuracy by less than 1% in either direction. My recommendation is to try a keras_tuner.Hyperband to see if a model can get above the goal of 75%. While features could be changed around to test for other possibilities for accuracy, the third attempt shows the worst accuracy score of them all so I would not recommend continuing down that venture.

