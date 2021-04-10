import numpy as np
import pandas as pd
from sklearn.datasets import load_iris



data = load_iris()
X, y, column_mames = data['data'], data['target'], data['feature_names']
#X = pd.DataFrame(X, columns=column_mames)

#nbc = nbc()
#nbc.fit(X,y)

print(X.items())
