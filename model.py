# import pandas for dataframe manipulation
import pandas as pd

# imports for visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# import for statistical analysis
from scipy import stats as stats

# import for modeling data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

# import for process repitition, aquiring, and preparing TelcoCo data
import acquire as aq
import prepare as prep


#---------------------------------------------------------------
# calling the aquire.py file get_telco_data() function
telco = aq.get_telco_data()

# calling the prepare.py file and storing the prepped data in an attribute
df = prep.prep_telco(telco)

# splitting the prepared data into my 3 working datasets
train, validate, test = df

#-------------------------------------------------------------
drop_columns = ['customer_id', 'churn', 'multiple_lines']

X_train = train.drop(columns = drop_columns)
y_train = train.churn

X_validate = validate.drop(columns = drop_columns)
y_validate = validate.churn

X_test = test.drop(columns = drop_columns)
y_test = test.churn
#----------------------------------------------------------------

def prepare_datasets():
    drop_columns = ['customer_id', 'churn', 'multiple_lines']

    X_train = train.drop(columns = drop_columns)
    y_train = train.churn

    X_validate = validate.drop(columns = drop_columns)
    y_validate = validate.churn

    X_test = test.drop(columns = drop_columns)
    y_test = test.churn

    print(f'X_train Shape: {X_train.shape}')
    print(f'X_validate Shape: {X_validate.shape}')
    print(f'X_test Shape: {X_test.shape}')


#-----------------------------------------------------------------
    
def forest_model():
    metrics = []
    max_depth = 12

    for i in range(1, max_depth):
        # Make the model
        depth = max_depth - i
        n_samples = i
        forest = RandomForestClassifier(max_depth=depth, min_samples_leaf=n_samples, random_state=123)

        # Fit the model (on train and only train)
        forest = forest.fit(X_train, y_train)

        # Use the model
        # We'll evaluate the model's performance on train, first
        in_sample_accuracy = forest.score(X_train, y_train)
        
        out_of_sample_accuracy = forest.score(X_validate, y_validate)

        output = {
            "min_samples_per_leaf": n_samples,
            "max_depth": depth,
            "train_accuracy": in_sample_accuracy,
            "validate_accuracy": out_of_sample_accuracy
        }
        
        metrics.append(output)
        
    df1 = pd.DataFrame(metrics)
    df1["difference"] = df1.train_accuracy - df1.validate_accuracy
    return df1







def knn_model():
    # Evaluate KNearest Neighbors models on train & validate set by looping through different values for k hyperparameter

    # create empty list for which to append scores from each loop
    scores = []
    k_range = range (1,11)
    # create loop for range 1-10
    for k in k_range:
                
        # define the model setting hyperparameters to values for current loop
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # fit the model on train
        knn.fit(X_train, y_train)
        
        # use the model and evaluate performance on train
        train_accuracy = knn.score(X_train, y_train)
        # use the model and evaluate performance on validate
        validate_accuracy = knn.score(X_validate, y_validate)
        
        # create output of current loop's hyperparameters and accuracy to append to metrics
        output = {
            "k": k,
            "train_accuracy": train_accuracy,
            "validate_accuracy": validate_accuracy
        }
        
        scores.append(output)

    # convert scores list to a dataframe for easy reading
    df2 = pd.DataFrame(scores)
    # add column to assess the difference between train & validate accuracy
    df2['difference'] = df2.train_accuracy - df2.validate_accuracy
    return df2






def logistic_regression_model():
    # Evaluate Logistic Regression models on train & validate set by looping through different values for c hyperparameter

    # create empty list for which to append metrics from each loop
    metrics = []

    # create loop for values in list
    for c in [.001, .005, .01, .05, .1, .5, 1, 5, 10, 50, 100, 500, 1000]:
                
        # define the model setting hyperparameters to values for current loop
        logit = LogisticRegression(C=c)
        
        # fit the model on train
        logit.fit(X_train, y_train)
        
        # use the model and evaluate performance on train
        train_accuracy = logit.score(X_train, y_train)
        # use the model and evaluate performance on validate
        validate_accuracy = logit.score(X_validate, y_validate)
        
        # create output of current loop's hyperparameters and accuracy to append to metrics
        output = {
            'C': c,
            'train_accuracy': train_accuracy,
            'validate_accuracy': validate_accuracy
        }
        
        metrics.append(output)

    # convert metrics list to a dataframe for easy reading
    df3 = pd.DataFrame(metrics)
    # add column to assess the difference between train & validate accuracy
    df3['difference'] = df3.train_accuracy - df3.validate_accuracy
    return df3




def test_dataset_accuracy():
    logit = LogisticRegression(C=1)

    #fit the data
    logit.fit(X_test, y_test)

    #score the data
    score = (logit.score(X_test, y_test)*100)
    score_percent = "{:.2f}".format(score)

    print(f'Test Dataset Accuracy Score : {score_percent}%')



def predictions_and_probability():
    logit = LogisticRegression(C=1)

    #fit the data
    logit.fit(X_test, y_test)

    #score the data
    score = (logit.score(X_test, y_test)*100)
    score_percent = "{:.2f}".format(score)

    y_prediction = logit.predict(X_test)
    y_prediction_probability = logit.predict_proba(X_test)
    predictions = pd.DataFrame(columns=['customer_id','probability_of_retention',
                                        'churn_prediction', 'actual_churn'])
    predictions['customer_id'] = test.customer_id
    predictions['probability_of_retention'] = y_prediction_probability
    predictions['churn_prediction'] = y_prediction
    predictions['actual_churn'] = test.churn

    predictions.churn_prediction = predictions.churn_prediction.map({1: 'Yes', 0: 'No'})
    predictions.actual_churn = predictions.actual_churn.map({1: 'Yes', 0: 'No'})

    return predictions
        
