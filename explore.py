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

# calling the aquire.py file get_telco_data() function
telco = aq.get_telco_data()

# calling the prepare.py file and storing the prepped data in an attribute
df = prep.prep_telco(telco)

# splitting the prepared data into my 3 working datasets
train, validate, test = df


def full_corr_chart():

    # calling the aquire.py file get_telco_data() function
    telco = aq.get_telco_data()

    # calling the prepare.py file and storing the prepped data in an attribute
    df = prep.prep_telco(telco)

    # splitting the prepared data into my 3 working datasets
    train, validate, test = df
    
    plt.figure(figsize= (15, 8))
    train.corr()['churn'].sort_values(ascending=False).plot(kind='barh',
                                                                 color = 'orange')
    plt.title('All features and their relation to churn', fontsize = 18)
    plt.xlabel('Correlation', fontsize = 18)
    plt.ylabel('Features', fontsize = 18)


def churn_corr_chart():
    # calling the aquire.py file get_telco_data() function
    telco = aq.get_telco_data()

    # calling the prepare.py file and storing the prepped data in an attribute
    df = prep.prep_telco(telco)

    # splitting the prepared data into my 3 working datasets
    train, validate, test = df
    
    train_corr = train.drop(['tenure','contract_type_Two year','no_internet',
                        'total_charges', 'one_year_contract', 'dependents', 'partner',
                        'DSL_internet', 'credit_card_pay', 'bank_transfer_pay', 
                        'mailed_check_pay', 'is_male'], axis = 1)
    
    plt.figure(figsize= (15, 8))
    train_corr.corr()['churn'].sort_values(ascending=False).plot(kind='barh',
                                                                 color = 'orange')
    plt.title('Drivers of churn', fontsize = 18)
    plt.xlabel('Correlation', fontsize = 18)
    plt.ylabel('Features', fontsize = 18)


def check_internet_types():
    # check internet service types from original telco data

    internet_types = ['DSL_internet', 'fiber_optic_internet', 'no_internet']

    for types in internet_types:
        sns.set_theme(style="whitegrid")
        ax = sns.countplot(x= train[types], hue="churn",
                           data=train)
        ax.set(xlabel =None)
        ax.set_xticklabels(['No', 'Yes'])
        plt.legend(['Has not churned', 'Has churned'])
        plt.title(f'Service: {types}')
        plt.show()

def check_payment_types():
    payment_types = ['bank_transfer_pay', 'credit_card_pay',
                 'electronic_check_pay', 'mailed_check_pay']

    for types in payment_types:
        sns.set_theme(style="whitegrid")
        ax = sns.countplot(x= train[types], hue="churn",
                           data=train)
        ax.set(xlabel =None)
        ax.set_xticklabels(['No', 'Yes'])
        plt.title(f'Payment Type: {types}')
        plt.show()

def dependents_and_tenure():
    ax3 = sns.catplot(x= 'dependents', y = 'tenure', hue="churn",
                       data=train, kind='bar')
    plt.title(f'Has Dependents')
    ax3.set_xticklabels(['No', 'Yes'])
    ax3.set(xlabel =None)
    plt.show()

def month_to_month_average_churn():
    # get all month to month contract holders
    m2m = train[train.month_to_month_contract == 1]

    # get the month to month contract holder who have churned
    m2m_churn = m2m[m2m.churn == 1]

    # get the average tenure of the month to month contract holders
    # who have churned

    m2m_churned_avg_tenure = m2m_churn.tenure.mean()

    sns.catplot(x='churn', y = 'tenure', kind='box', size=10,
               data = train[train.churn == 1])

    plt.title('The Average Churn of Month to Month Telco Customers')

    plt.xlabel('Churned', fontsize = 14)

    plt.ylabel('Months with Telco Services')

    plt.axhline(y= m2m_churned_avg_tenure,
                linestyle = '--', color = 'red')

    plt.figure(figsize=(10,8))

    sns.histplot(data=train, x='tenure', hue='churn',
                 palette = 'flare')

    plt.legend(['No', 'Yes'])
    plt.xlabel('Tenure in Months', fontsize=13)
    plt.ylabel('Customer Count', fontsize = 13)
