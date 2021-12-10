import pandas as pd
import numpy as np
from genetic import Genome
from random import choices, randint, randrange, random
from randomForest import randomForest

df=pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')


#Remove the column EmployeeNumber
df = df.drop('EmployeeNumber', axis = 1) # A number assignment
#Remove the column StandardHours
df = df.drop('StandardHours', axis = 1) #Contains only value 80
#Remove the column EmployeeCount
df = df.drop('EmployeeCount', axis = 1) #Contains only the value 1
#Remove the column EmployeeCount
df = df.drop('Over18', axis = 1) #Contains only the value 'Yes'

from sklearn.preprocessing import LabelEncoder

for column in df.columns:
        if df[column].dtype == np.number:
            continue
        df[column] = LabelEncoder().fit_transform(df[column])

#Create a new column at the end of the dataframe that contains the same value
df['Age_Years'] = df['Age']
#Remove the first column called age
df = df.drop('Age', axis = 1)
X = df.iloc[:, 1:df.shape[1]].values
Y = df.iloc[:, 0].values


# def generate_genome(length: int) -> Genome:
#     return choices([0, 1], k=length)


features=[e for e in df.keys()][1:]


def fitness(genome: Genome,feature_limit: int=15):
    c=0
    selected_features=[]
    for choice in genome:
        c+=choice
    if c>feature_limit:
        return 0
    else:
        for i,choice in enumerate(genome):
            if choice==1:
                selected_features.append(i)
        return randomForest(X[:,selected_features],Y)







