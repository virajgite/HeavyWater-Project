# HeavyWater-Project

**Project aim:**

Train a document classification model. Deploy the model to a public cloud platform.

**Data:**

Contains two columns - document label and hashed document content 

Original Document Distribution 

![original data distribution](https://github.com/virajgite/HeavyWater-Project/blob/master/Screenshot%20(189).png)

**Total Documents:62204**


Since the document distribution is uneven I balance the dataset by undersampling, which reduces the number of documents to be preocessed and computation power required

Undersampled Document Distribution

![undersampled data distribution](https://github.com/virajgite/HeavyWater-Project/blob/master/Screenshot%20(190).png)

**Total Documents:3206**


**Model:**

I used (Multinomial) Naive Bayes model available in the scikit-learn. Trained the document on the undersampled document set and tested the model with various different sizes of test data. 

When splitting the undersampled document set into 80%-train 20%-test, the model predicts the test data with 75% accuracy. For larger test data sizes the model consistently provides an accurracy of 75%.

It is important to test different ML models(like Neural Nets, Random Forest etc) when devising a solution to a problem using ML. I did not check other models due to time constraint.

**Deployment:**

The prediction function is deployed on AWS lambda, and the model and required packages for the functions are stored on AWS S3.

**Steps Taken:**

1. Undersampling Data
2. Creating of features vectors using tfidf vectorizer
3. Training the model on undersampled data
4. Saving the model and feature vocabulary to be used in the prediction function
5. Deploying the function to AWS lamda 
(Deploying a function to lambda involves building required libraries for prediction function from source in the AWS linux environment. A docker image of AWS linux env can be downloaded [here](https://hub.docker.com/_/amazonlinux/). The built libraries and lamda_function need to be uploaded on S3 as a .zip file) 
6. Creating API on AWS API GATEWAY
7. Building an UI for submitting requests to the API (uses Django)


**Improvements**

1. The classification can be improved by removing stop-words before hashing the raw text. 
2. A list of stop-words can be formed in context with the domain in which this ML model will be used, I believe many stop-words might exist in financial documents that barely helps to distinguish between different classes of documents. 
3. Trying multiple models and parameter tuning will help us to use the best suitable model.

**scikit-learn package built for AWS lambda (amazonlinux2018.03)** [download](https://drive.google.com/file/d/1PjOkB3LFyrNPi-pb80ObD3n2ckQ5o-vB/view?usp=sharing)

_**[Contact me](mailto:virajgite@gmail.com) for links to the UI and API**_



