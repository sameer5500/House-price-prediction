'''Copyright (c) 2022 AIClub

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the "Software"), to deal in the Software without restriction, including without 
limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of 
the Software, and to permit persons to whom the Software is furnished to do so, subject to the following 
conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO 
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN 
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE 
OR OTHER DEALINGS IN THE SOFTWARE.

Follow our courses - https://www.corp.aiclub.world/courses'''

def launch_fe(data):
    import os
    import pandas as pd
    import numpy as np
    from io import StringIO
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction import text
    import pickle
    from scipy import sparse
    MAX_TEXT_FEATURES = 200
    columns_list = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "households", "median_income", "population", "median_house_value"]

    dataset = pd.read_csv(data, skipinitialspace=True)

    # Replace inf and -inf, with max and min values of the particular column
    df = dataset.select_dtypes(include=np.number)
    cols = df.columns.to_series()[np.isinf(df).any()]
    col_min_max = {np.inf: dataset[cols][np.isfinite(dataset[cols])].max(), -np.inf: dataset[cols][np.isfinite(dataset[cols])].min()} 
    dataset[cols] = dataset[cols].replace({col: col_min_max for col in cols})

    num_samples = len(dataset)

    # Move the label column
    cols = list(dataset.columns)
    colIdx = dataset.columns.get_loc("median_house_value")
    # Do nothing if the label is in the 0th position
    # Otherwise, change the order of columns to move label to 0th position
    if colIdx != 0:
        cols = cols[colIdx:colIdx+1] + cols[0:colIdx] + cols[colIdx+1:]
        dataset = dataset[cols]

    # split dataset into train and test
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)

    # Write train and test csv
    train.to_csv('train.csv', index=False, header=False)
    test.to_csv('test.csv', index=False, header=False)
    column_names = list(train.columns)
def get_model_id():
    return "None"

# Please replace the brackets below with the location of your data file
data = '<>'

launch_fe(data)

# import the library of the algorithm
from sklearn.linear_model import LinearRegression

# Initialize hyperparams
fit_intercept = True

# Initialize the algorithm
model = LinearRegression(fit_intercept=fit_intercept)
algorithm = 'LinearRegression'

import pandas as pd
# Load the test and train datasets
train = pd.read_csv('train.csv', skipinitialspace=True, header=None)
test = pd.read_csv('test.csv', skipinitialspace=True, header=None)
# Train the algorithm
model.fit(train.iloc[:,1:], train.iloc[:,0])

import numpy as np
# Predict the target values
y_pred = model.predict(test.iloc[:, 1:])
# calculate rmse
rmse = np.sqrt(np.mean((y_pred - test.iloc[:, 0])**2))
print('RMSE of the model is: ', rmse)
# import the library to calculate mae
from sklearn.metrics import mean_absolute_error
# calculate mae
mae = mean_absolute_error(np.array(test.iloc[:, 0]), y_pred)
print('MAE of the model is: ', mae)

# fe_transform function traansforms raw data into a form the model can consume
print('Below is the prediction stage of the AI')
def fe_transform(data_dict, object_path=None):
    import os
    import pandas as pd
    from io import StringIO
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction import text
    import pickle
    from scipy import sparse
    
    dataset = pd.DataFrame([data_dict])

    return dataset

test_sample = {'longitude': -119.33, 'latitude': 37.245000000000005, 'housing_median_age': 26.5, 'total_rooms': 19661.0, 'total_bedrooms': 3223.0, 'households': 3041.5, 'median_income': 7.75, 'population': 17842.5}
# Call FE on test_sample
test_sample_modified = fe_transform(test_sample)
# Make a prediction
prediction = model.predict(test_sample_modified)
print(prediction)
