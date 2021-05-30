import numpy as np #Handling data with np arrays, other algebraic manipulations
import pandas as pd #Data Processing

#For Pre-processing
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler

#Models which we will try
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier


# Validation libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def initialize():
    #Reading the survey dataset file
    df = pd.read_csv(r'C:\Users\marka\Desktop\Btech_project\Code\Django\testquiz\testapp1\survey.csv')

    #We had some rows where age was clearly wrong and had values like 999999, -175
    #Lets replace these values with 0 so that later we can fill them with mean
    df.loc[(df['Age'] >= 100) | (df['Age'] < 0), 'Age'] = 0


    #We have certain attributes in our dataset like timestamp, comments and state
    # (US states only) which are not important for our project. So we are discarding them
    df = df.drop(['Timestamp', 'state', 'comments'], axis=1)

    #In all the string type attributes like 'Gender', 'Country', 'self_employed' etc. 
    # we need to put NaN wherever we have null-type data

    stringAttributes = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                    'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                    'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                    'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                    'seek_help']

    for attribute in df:
        if(attribute in stringAttributes):
            df[attribute] = df[attribute].fillna('NaN')

            df.drop(df[df[attribute]==31].index, inplace=False)

    for attribute in stringAttributes:
        df.drop(df[df[attribute]==31].index, inplace=True)

    #In the survey.csv file notice that the gender attribute has a lot of problems
    # for male the entries include m, M, Male, male, MaLE etc. We need to standardize the,
    # as male; Similarly for female

    #  furthermore there are a lot more gender groups who don't identify as male or female
    #  such as tras, queer, fluid, non-binary. Let's standardize them as trans

    # Gender Groups
    male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]

    trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]  

    female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

    for (index, row) in df.iterrows():
        if(str.lower(row['Gender']) in male_str):
            df.loc[index, 'Gender'] = 'male'
        elif(str.lower(row['Gender']) in female_str):
            df.loc[index, 'Gender'] = 'female'
        else:
            df.loc[index, 'Gender'] = 'trans'

    # Fill weird data in age with median of age
    df.loc[(df['Age'] < 18) | (df['Age'] == 0)] = df['Age'].median()

    # Make ranges of age
    # Add new attribute age_range to df
    df['age_range'] = pd.cut(df['Age'], [0,20,30,65,100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)

    # In self_employed attribute change NaN to Not self-employed
    df['self_employed'] = df['self_employed'].replace(['NaN'], 'No')
    # In work_interfere change NaN or 31 to Don't know
    df['work_interfere'] = df['work_interfere'].replace(['NaN', 31], f"Don't know")
    for attribute in stringAttributes:
        df.drop(df[df[attribute]==31].index, inplace=True)

    # Observe that since data points are more for USA, Canada and Australia treatment count is high in these countries. To avoid skewed data and being country specific, let's get rid of country attribute.

    df = df.drop(['Country'], axis=1)

    labelDict = {}
    for feature in df:
        le = preprocessing.LabelEncoder()
        le.fit(df[feature])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        df[feature] = le.transform(df[feature])
        
        # Get labels
        labelKey = 'label_' + feature
        labelValue = [*le_name_mapping]
        labelDict[labelKey] = labelValue
        
    # Scaling Age


    scaler = MinMaxScaler()
    df['Age'] = scaler.fit_transform(df[['Age']])


    # We need to define X and y as independent and dependent attributes
    # Attributes in X will help us to predict attribute y
    # In our case independent attributes are listed below and we will be predicting the value of "treatment" attribute.


    independent_attributes = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']

    X = df[independent_attributes]
    y = df['treatment']

    # Let's split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    return [df, X_train, y_train, X_test, y_test]


# ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']

test_tup = [[0.33, 1, 0, 1, 2, 0, 0, 0]]
# age = 33 yrs, gender = male, family hist of mental illness = no, benefits=no, careoptions=yes, anon=dk, leave=dk, wi=dk
another_test_tup = [[0.67, 0, 1, 1, 0, 0, 3, 2]]
# age = 67 yrs, gender = male, family hist of mental illness = yes, benefits=no, careoptions=no, anon=dk, leave=very difficult, wi=yes
# 0.431818	0	0	2	1	2	2	2
# 0.636364	0	0	1	2	0	0	0



def RF(test_tup, X_train, y_train, X_test):
    forest = RandomForestClassifier(max_depth = None, min_samples_leaf=8, min_samples_split=2, n_estimators = 31, random_state = 1)
    forest.fit(X_train, y_train)

    y_pred_class = forest.predict(X_test)

    result = [forest.predict(test_tup), "{:.2f}".format(forest.predict_proba(test_tup)[0][1]), "{:.2f}".format(forest.score(X_train,y_train))]

    # if(forest.predict(test_tup)==[1]):
    #     result = f'Random Forest has predicted that you should seek mental treatment based on the inputs given for workplace factors. \nThe probability value for seeking treatment came out to be {"{:.2f}".format(forest.predict_proba(test_tup)[0][1])} with RF giving an accuracy of {"{:.2f}".format(forest.score(X_train,y_train))}'
    # else:
    #     result = f'Random Forest has predicted that you might not have to seek mental treatment based on the inputs given for workplace factors. The probability value for seeking treatment came out to be {"{:.2f}".format(forest.predict_proba(test_tup)[0][1])} with RF giving an accuracy of {"{:.2f}".format(forest.score(X_train,y_train))}'

    return result

def executeRF(another_test_tup):
    print(another_test_tup)
    df, X_train, y_train, X_test, y_test = initialize()
    res = RF(another_test_tup, X_train, y_train, X_test)
    print(res)
    return(res)

# executeRF(test_tup)