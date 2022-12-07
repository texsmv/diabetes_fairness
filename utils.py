import pandas as pd
import numpy as np


def read_diabetes_dataset(binary = False):
    # Read the dataset
    data = pd.read_csv('diabetic_data.csv')
    
    
    # Remove duplicates based on patient_nbr
    
    data = data.drop_duplicates(subset=['patient_nbr'])
    
    
    # Remove Uninformative Features
    # the uninformative features in the dataset (21 in total) were discarded as shown in the table below, due to either, a huge amount of missing sample values (>50%), or due to the fact that somefeatures are not relevant to classify the data towards our target (Like patient ID), or if the feature is compeletly unbalanced (>95% of data points have the same value for the feature).
    
    features_drop_list = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'repaglinide', 'nateglinide', 'chlorpropamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone','metformin-pioglitazone', 'acetohexamide', 'tolbutamide']
    data = data.drop(features_drop_list, axis=1)
    
    
    # Remove nan values 
    data = data.replace(" ?", np.nan)
    data = data.dropna().reset_index(drop=True)
    
    
    
    #start by setting all values containing E or V into 0 (as one category)
    data.loc[data['diag_1'].str.contains('V',na=False,case=False), 'diag_1'] = 0
    data.loc[data['diag_1'].str.contains('E',na=False,case=False), 'diag_1'] = 0
    data.loc[data['diag_2'].str.contains('V',na=False,case=False), 'diag_2'] = 0
    data.loc[data['diag_2'].str.contains('E',na=False,case=False), 'diag_2'] = 0
    data.loc[data['diag_3'].str.contains('V',na=False,case=False), 'diag_3'] = 0
    data.loc[data['diag_3'].str.contains('E',na=False,case=False), 'diag_3'] = 0

    #setting all missing values into -1
    data['diag_1'] = data['diag_1'].replace('?', -1)
    data['diag_2'] = data['diag_2'].replace('?', -1)
    data['diag_3'] = data['diag_3'].replace('?', -1)

    #No all diag values can be converted into numeric values
    data['diag_1'] = data['diag_1'].astype(float)
    data['diag_2'] = data['diag_2'].astype(float)
    data['diag_3'] = data['diag_3'].astype(float)

    

    #Now we will reduce the number of categories in diag features according to ICD-9 code
    #(Missing values will be grouped as E & V values)
    data['diag_1'].loc[(data['diag_1']>=1) & (data['diag_1']< 140)] = 1
    data['diag_1'].loc[(data['diag_1']>=140) & (data['diag_1']< 240)] = 2
    data['diag_1'].loc[(data['diag_1']>=240) & (data['diag_1']< 280)] = 3
    data['diag_1'].loc[(data['diag_1']>=280) & (data['diag_1']< 290)] = 4
    data['diag_1'].loc[(data['diag_1']>=290) & (data['diag_1']< 320)] = 5
    data['diag_1'].loc[(data['diag_1']>=320) & (data['diag_1']< 390)] = 6
    data['diag_1'].loc[(data['diag_1']>=390) & (data['diag_1']< 460)] = 7
    data['diag_1'].loc[(data['diag_1']>=460) & (data['diag_1']< 520)] = 8
    data['diag_1'].loc[(data['diag_1']>=520) & (data['diag_1']< 580)] = 9
    data['diag_1'].loc[(data['diag_1']>=580) & (data['diag_1']< 630)] = 10
    data['diag_1'].loc[(data['diag_1']>=630) & (data['diag_1']< 680)] = 11
    data['diag_1'].loc[(data['diag_1']>=680) & (data['diag_1']< 710)] = 12
    data['diag_1'].loc[(data['diag_1']>=710) & (data['diag_1']< 740)] = 13
    data['diag_1'].loc[(data['diag_1']>=740) & (data['diag_1']< 760)] = 14
    data['diag_1'].loc[(data['diag_1']>=760) & (data['diag_1']< 780)] = 15
    data['diag_1'].loc[(data['diag_1']>=780) & (data['diag_1']< 800)] = 16
    data['diag_1'].loc[(data['diag_1']>=800) & (data['diag_1']< 1000)] = 17
    data['diag_1'].loc[(data['diag_1']==-1)] = 0

    data['diag_2'].loc[(data['diag_2']>=1) & (data['diag_2']< 140)] = 1
    data['diag_2'].loc[(data['diag_2']>=140) & (data['diag_2']< 240)] = 2
    data['diag_2'].loc[(data['diag_2']>=240) & (data['diag_2']< 280)] = 3
    data['diag_2'].loc[(data['diag_2']>=280) & (data['diag_2']< 290)] = 4
    data['diag_2'].loc[(data['diag_2']>=290) & (data['diag_2']< 320)] = 5
    data['diag_2'].loc[(data['diag_2']>=320) & (data['diag_2']< 390)] = 6
    data['diag_2'].loc[(data['diag_2']>=390) & (data['diag_2']< 460)] = 7
    data['diag_2'].loc[(data['diag_2']>=460) & (data['diag_2']< 520)] = 8
    data['diag_2'].loc[(data['diag_2']>=520) & (data['diag_2']< 580)] = 9
    data['diag_2'].loc[(data['diag_2']>=580) & (data['diag_2']< 630)] = 10
    data['diag_2'].loc[(data['diag_2']>=630) & (data['diag_2']< 680)] = 11
    data['diag_2'].loc[(data['diag_2']>=680) & (data['diag_2']< 710)] = 12
    data['diag_2'].loc[(data['diag_2']>=710) & (data['diag_2']< 740)] = 13
    data['diag_2'].loc[(data['diag_2']>=740) & (data['diag_2']< 760)] = 14
    data['diag_2'].loc[(data['diag_2']>=760) & (data['diag_2']< 780)] = 15
    data['diag_2'].loc[(data['diag_2']>=780) & (data['diag_2']< 800)] = 16
    data['diag_2'].loc[(data['diag_2']>=800) & (data['diag_2']< 1000)] = 17
    data['diag_2'].loc[(data['diag_2']==-1)] = 0

    data['diag_3'].loc[(data['diag_3']>=1) & (data['diag_3']< 140)] = 1
    data['diag_3'].loc[(data['diag_3']>=140) & (data['diag_3']< 240)] = 2
    data['diag_3'].loc[(data['diag_3']>=240) & (data['diag_3']< 280)] = 3
    data['diag_3'].loc[(data['diag_3']>=280) & (data['diag_3']< 290)] = 4
    data['diag_3'].loc[(data['diag_3']>=290) & (data['diag_3']< 320)] = 5
    data['diag_3'].loc[(data['diag_3']>=320) & (data['diag_3']< 390)] = 6
    data['diag_3'].loc[(data['diag_3']>=390) & (data['diag_3']< 460)] = 7
    data['diag_3'].loc[(data['diag_3']>=460) & (data['diag_3']< 520)] = 8
    data['diag_3'].loc[(data['diag_3']>=520) & (data['diag_3']< 580)] = 9
    data['diag_3'].loc[(data['diag_3']>=580) & (data['diag_3']< 630)] = 10
    data['diag_3'].loc[(data['diag_3']>=630) & (data['diag_3']< 680)] = 11
    data['diag_3'].loc[(data['diag_3']>=680) & (data['diag_3']< 710)] = 12
    data['diag_3'].loc[(data['diag_3']>=710) & (data['diag_3']< 740)] = 13
    data['diag_3'].loc[(data['diag_3']>=740) & (data['diag_3']< 760)] = 14
    data['diag_3'].loc[(data['diag_3']>=760) & (data['diag_3']< 780)] = 15
    data['diag_3'].loc[(data['diag_3']>=780) & (data['diag_3']< 800)] = 16
    data['diag_3'].loc[(data['diag_3']>=800) & (data['diag_3']< 1000)] = 17
    data['diag_3'].loc[(data['diag_3']==-1)] = 0



    

    data['race'] = data['race'].replace('?', 'Other')


    # Here we have just 3 values that are Invalid, as a result we will change them into Female, which is the mod for this feature.
    data['gender'] = data['gender'].replace('Unknown/Invalid', 'Female')
    
    

    # Now we can easliy change Male/Female into 1/0
    data['gender'] = data['gender'].replace('Male', 1)
    data['gender'] = data['gender'].replace('Female', 0)


    # For age, we have 10 categories, each represents 10 years range from [0-10] to [90-100]. We will replace those with the middle age for each age range: for example (0,10] will be repleased with 5; (60, 70] will be replaces by 65; and so on.
    for i in range(0,10):
        data['age'] = data['age'].replace('['+str(10*i)+'-'+str(10*(i+1))+')', i*10+5)

     
    # Replace by 4 numerical categories
    data['max_glu_serum']=data['max_glu_serum'].replace("None", 0)
    data['max_glu_serum']=data['max_glu_serum'].replace("Norm", 1)
    data['max_glu_serum']=data['max_glu_serum'].replace(">200", 2)
    data['max_glu_serum']=data['max_glu_serum'].replace(">300", 3)
    
    
    # Replace by 4 numerical categories
    data['A1Cresult']=data['A1Cresult'].replace("None", 0)
    data['A1Cresult']=data['A1Cresult'].replace("Norm", 1)
    data['A1Cresult']=data['A1Cresult'].replace(">7", 2)
    data['A1Cresult']=data['A1Cresult'].replace(">8", 3)



    # All values in :    metformin,     glimepiride,    glipizide,    glyburide,    pioglitazone,    rosiglitazone,    insulin
    # can be No, Dowm, Steady and Up ...  replace them by 4 values
    drug_list = ['metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin']
    for i in drug_list:
        data[i] = data[i].replace('No', 0)
        data[i] = data[i].replace('Steady', 2)
        data[i] = data[i].replace('Down', 1)
        data[i] = data[i].replace('Up', 3)


    # Convert change into binary representation
    data['change']=data['change'].replace('No', 0)
    data['change']=data['change'].replace('Ch', 1)
     
    
    # Convert diabetesMed into binary representation
    data['diabetesMed']=data['diabetesMed'].replace('Yes', 1)
    data['diabetesMed']=data['diabetesMed'].replace('No', 0)
    
    # Race into binary categories
    data = pd.concat([data,pd.get_dummies(data['race'], prefix='race')], axis=1).drop(['race'],axis=1)


    # Convert readmitted into numerical representation
    if binary:
        data['readmitted']=data['readmitted'].replace('NO', 0)
        data['readmitted']=data['readmitted'].replace('>30', 1)
        data['readmitted']=data['readmitted'].replace('<30', 1)
    else:
        data['readmitted']=data['readmitted'].replace('NO', 0)
        data['readmitted']=data['readmitted'].replace('>30', 1)
        data['readmitted']=data['readmitted'].replace('<30', 2)


    # Divide labels and features
    y = data['readmitted']
    X = data.drop(['readmitted'], axis=1)
    
    return X, y
