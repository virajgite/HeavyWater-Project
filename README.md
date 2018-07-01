# HeavyWater-Project

Project aim: 
Train a document classification model. Deploy the model to a public cloud platform.

Data:
Contains two columns - document label and hashed document content 

Original Document Distribution 

|BILL|18968|
POLICY CHANGE|10627
CANCELLATION NOTICE|9731
BINDER|8973
DELETION OF INTEREST|4826
REINSTATEMENT NOTICE|4368
DECLARATION|968
CHANGE ENDORSEMENT|889
RETURNED CHECK|749
EXPIRATION NOTICE|734
NON-RENEWAL NOTICE|624
BILL BINDER|289
APPLICATION|229
INTENT TO CANCEL NOTICE|229

Total Documents|62204


Since the document distribution is uneven I balance the dataset by undersampling which reduces the number of documents to be preocessed and computation power required

Undersampled Document Distribution
CHANGE ENDORSEMENT         229
RETURNED CHECK             229
BINDER                     229
BILL                       229
DECLARATION                229
EXPIRATION NOTICE          229
NON-RENEWAL NOTICE         229
REINSTATEMENT NOTICE       229
DELETION OF INTEREST       229
POLICY CHANGE              229
APPLICATION                229
CANCELLATION NOTICE        229
BILL BINDER                229
INTENT TO CANCEL NOTICE    229

Model:
I used (Multinomial) Naive Bayes model available in the scikit-learn. Trained the document on the undersampled document set and tested the model with various different sizes of test data. When splitting the undersampled document-set into 80% train 20% test. the model predicts the test data with 75% accuracy. For larger test data sizes consistently provides an accurracy of 75%.

While it is important to test different ML models(like Neural Nets, Random Forest etc) when devising a solution to a problem using ML, I did not check other models due to time constraint.

Deployment:
The preiction function is deployed on AWS lambda, and the model and required packages for the functions are stored on AWS S3.

Steps Taken:
1. Undersampling Data
2. Creating of features vectors using tfidf vectorizer
3. Training the model on undersampled data
4. Saving the model and feature vocabulary to be used in the prediction function
5. Deploying the function to AWS lamda 
(Deploying a function to lambda involves building required libraries for prediction function from source in the AWS linux environment. A docker image of AWS linux env can be downloaded [here](https://hub.docker.com/_/amazonlinux/). The built libraries and lamda_function need to be uploaded on S3 as a .zip file) 
6.Creating API on AWS API GATEWAY
7.Building an UI for submitting requests to the API (uses Django)

Contact me for links to the UI and API



