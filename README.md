# ML_Regression_Project
Machine Learning Project to make regression prediction for housing price
There are two datasets for models; Boston Housing price and Melbourne housing price
For Boston Housing price, 
```python
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd

boston = load_boston()
dfX = pd.DataFrame(boston.data, columns= boston.feature_names)
dfy = pd.DataFrame(boston.target, columns=["MEDV"])
df= pd.concat([dfX, dfy], axis=1)

N=len(df)
ratio=0.7
np.random.seed(0)
idx_train = np.random.choice(np.arange(N), np.int(ratio*N))
idx_test = list(set(np.arange(N)).difference(idx_train))

df_train = df.iloc[idx_train]
df_test = df.iloc[idx_test]
```

For Melbourne Housing Price, use the attatched melb_data.csv
