import os
import ctypes
import pickle
import botocore
import boto3
import json

s3 = boto3.resource('s3',aws_access_key_id='security concerns',
         aws_secret_access_key= 'security concerns')

bucket=s3.Bucket('heavywaterdata')
exists = True


for d, _, files in os.walk('lib'):
    for f in files:
        if f.endswith('.a'):
            continue
        ctypes.cdll.LoadLibrary(os.path.join(d, f))


from sklearn.feature_extraction.text import TfidfVectorizer

def lambda_handler(event,context):
    try:
        s3.meta.client.head_bucket(Bucket='heavywaterdata')
    except botocore.exceptions.ClientError as e:
    # If a client error is thrown, then check that it was a 404 error.
    # If it was a 404 error, then the bucket does not exist.
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            exists = False
        
    #for bucket in s3.buckets.all():
        #for key in bucket.objects.all():
            #print(key.key)
    
    #download model if not available locally 
    if os.path.exists('/tmp/'+'model_file.pkl'):
        print('not downloading model')
    else:
        print('downloading model')
        with open('/tmp/'+'model_file.pkl', 'wb') as data:
            bucket.download_fileobj('clf_model.pkl', data)
        
    with open('/tmp/'+'model_file.pkl', 'rb') as pickle_file:
        imported_model = pickle.load(pickle_file)
    
    #download feature/vocabulary if not locally available
    if os.path.exists('/tmp/'+'features_file.pkl'):
        print('not downloading features')
    else:
        print('downloading features')
        with open('/tmp/'+'features_file.pkl', 'wb') as data2:
            bucket.download_fileobj('features.pkl', data2)
    
    with open('/tmp/'+'features_file.pkl', 'rb') as pickle_file2:
        vocab_content = pickle.load(pickle_file2)
    
    #fetch input document 
    input_document=event["queryStringParameters"]["words"]
    #input_document=''
    
    #create a tfidf vectorizer with the vocabulary which was used when training
    tfidf_vectorizer=TfidfVectorizer(decode_error="replace",vocabulary=vocab_content)
    
    #vectorize the input document
    test_features=tfidf_vectorizer.fit_transform([input_document]).toarray()
    
    #predict the input document class 
    prediction=imported_model.predict(test_features)
    
    #clean the prediction string before returning
    prediction=str(prediction)
    prediction=prediction.strip("[")
    prediction=prediction.strip("]")
    prediction=prediction.strip("u")
    prediction=prediction.strip("'")
    
    #format output as required by AWS lambda in proxy configuration
    body_json={}
    body_json["prediction"]=prediction
    result={}
    result["isBase64Encoded"]=False
    result["statusCode"]=200
    #result["headers"]={"Content-Type":"application/json"}
    result["body"]=json.dumps(body_json)
    
    return result