import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv("PATH",header=None)
df[0].value_counts()

rus=RandomUnderSampler(random_state=0,return_indices=True)

Xarray=df[1].values.astype(str)

features_sampled,labels_sampled,indices=rus.fit_sample(Xarray.reshape(-1,1),df[0])

output_array=np.concatenate((labels_sampled,features_sampled),axis=1)

np.savetxt('sampled_with_names.csv',output_array,fmt='%s',delimiter=',')