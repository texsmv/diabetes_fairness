import pandas as pd
import numpy as np
import itertools

from sklearn.base import BaseEstimator, MetaEstimatorMixin


def read_diabetes_dataset(binary=False):
    # Read the dataset
    data = pd.read_csv('diabetic_data.csv')

    # Remove duplicates based on patient_nbr
    data = data.drop_duplicates(subset=['patient_nbr'])

    # Remove Uninformative Features
    # the uninformative features in the dataset (21 in total) were discarded as shown in the table below, due to either, a huge amount of missing sample values (>50%), or due to the fact that somefeatures are not relevant to classify the data towards our target (Like patient ID), or if the feature is compeletly unbalanced (>95% of data points have the same value for the feature).
    features_drop_list = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'repaglinide', 'nateglinide', 'chlorpropamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
                          'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'acetohexamide', 'tolbutamide']
    data = data.drop(features_drop_list, axis=1)

    # Remove nan values
    data = data.replace(" ?", np.nan)
    data = data.dropna().reset_index(drop=True)

    # start by setting all values containing E or V into 0 (as one category)
    data.loc[data['diag_1'].str.contains(
        'V', na=False, case=False), 'diag_1'] = 0
    data.loc[data['diag_1'].str.contains(
        'E', na=False, case=False), 'diag_1'] = 0
    data.loc[data['diag_2'].str.contains(
        'V', na=False, case=False), 'diag_2'] = 0
    data.loc[data['diag_2'].str.contains(
        'E', na=False, case=False), 'diag_2'] = 0
    data.loc[data['diag_3'].str.contains(
        'V', na=False, case=False), 'diag_3'] = 0
    data.loc[data['diag_3'].str.contains(
        'E', na=False, case=False), 'diag_3'] = 0

    # setting all missing values into -1
    data['diag_1'] = data['diag_1'].replace('?', -1)
    data['diag_2'] = data['diag_2'].replace('?', -1)
    data['diag_3'] = data['diag_3'].replace('?', -1)

    # No all diag values can be converted into numeric values
    data['diag_1'] = data['diag_1'].astype(float)
    data['diag_2'] = data['diag_2'].astype(float)
    data['diag_3'] = data['diag_3'].astype(float)

    # Now we will reduce the number of categories in diag features according to ICD-9 code
    # (Missing values will be grouped as E & V values)
    data.loc[(data['diag_1'] >= 1) & (data['diag_1'] < 140), 'diag_1'] = 1
    data.loc[(data['diag_1'] >= 140) & (data['diag_1'] < 240), 'diag_1'] = 2
    data.loc[(data['diag_1'] >= 240) & (data['diag_1'] < 280), 'diag_1'] = 3
    data.loc[(data['diag_1'] >= 280) & (data['diag_1'] < 290), 'diag_1'] = 4
    data.loc[(data['diag_1'] >= 290) & (data['diag_1'] < 320), 'diag_1'] = 5
    data.loc[(data['diag_1'] >= 320) & (data['diag_1'] < 390), 'diag_1'] = 6
    data.loc[(data['diag_1'] >= 390) & (data['diag_1'] < 460), 'diag_1'] = 7
    data.loc[(data['diag_1'] >= 460) & (data['diag_1'] < 520), 'diag_1'] = 8
    data.loc[(data['diag_1'] >= 520) & (data['diag_1'] < 580), 'diag_1'] = 9
    data.loc[(data['diag_1'] >= 580) & (data['diag_1'] < 630), 'diag_1'] = 10
    data.loc[(data['diag_1'] >= 630) & (data['diag_1'] < 680), 'diag_1'] = 11
    data.loc[(data['diag_1'] >= 680) & (data['diag_1'] < 710), 'diag_1'] = 12
    data.loc[(data['diag_1'] >= 710) & (data['diag_1'] < 740), 'diag_1'] = 13
    data.loc[(data['diag_1'] >= 740) & (data['diag_1'] < 760), 'diag_1'] = 14
    data.loc[(data['diag_1'] >= 760) & (data['diag_1'] < 780), 'diag_1'] = 15
    data.loc[(data['diag_1'] >= 780) & (data['diag_1'] < 800), 'diag_1'] = 16
    data.loc[(data['diag_1'] >= 800) & (data['diag_1'] < 1000), 'diag_1'] = 17
    data.loc[(data['diag_1'] == -1), 'diag_1'] = 0

    data.loc[(data['diag_2'] >= 1) & (data['diag_2'] < 140), 'diag_2'] = 1
    data.loc[(data['diag_2'] >= 140) & (data['diag_2'] < 240), 'diag_2'] = 2
    data.loc[(data['diag_2'] >= 240) & (data['diag_2'] < 280), 'diag_2'] = 3
    data.loc[(data['diag_2'] >= 280) & (data['diag_2'] < 290), 'diag_2'] = 4
    data.loc[(data['diag_2'] >= 290) & (data['diag_2'] < 320), 'diag_2'] = 5
    data.loc[(data['diag_2'] >= 320) & (data['diag_2'] < 390), 'diag_2'] = 6
    data.loc[(data['diag_2'] >= 390) & (data['diag_2'] < 460), 'diag_2'] = 7
    data.loc[(data['diag_2'] >= 460) & (data['diag_2'] < 520), 'diag_2'] = 8
    data.loc[(data['diag_2'] >= 520) & (data['diag_2'] < 580), 'diag_2'] = 9
    data.loc[(data['diag_2'] >= 580) & (data['diag_2'] < 630), 'diag_2'] = 10
    data.loc[(data['diag_2'] >= 630) & (data['diag_2'] < 680), 'diag_2'] = 11
    data.loc[(data['diag_2'] >= 680) & (data['diag_2'] < 710), 'diag_2'] = 12
    data.loc[(data['diag_2'] >= 710) & (data['diag_2'] < 740), 'diag_2'] = 13
    data.loc[(data['diag_2'] >= 740) & (data['diag_2'] < 760), 'diag_2'] = 14
    data.loc[(data['diag_2'] >= 760) & (data['diag_2'] < 780), 'diag_2'] = 15
    data.loc[(data['diag_2'] >= 780) & (data['diag_2'] < 800), 'diag_2'] = 16
    data.loc[(data['diag_2'] >= 800) & (data['diag_2'] < 1000), 'diag_2'] = 17
    data.loc[(data['diag_2'] == -1), 'diag_2'] = 0

    data.loc[(data['diag_3'] >= 1) & (data['diag_3'] < 140), 'diag_3'] = 1
    data.loc[(data['diag_3'] >= 140) & (data['diag_3'] < 240), 'diag_3'] = 2
    data.loc[(data['diag_3'] >= 240) & (data['diag_3'] < 280), 'diag_3'] = 3
    data.loc[(data['diag_3'] >= 280) & (data['diag_3'] < 290), 'diag_3'] = 4
    data.loc[(data['diag_3'] >= 290) & (data['diag_3'] < 320), 'diag_3'] = 5
    data.loc[(data['diag_3'] >= 320) & (data['diag_3'] < 390), 'diag_3'] = 6
    data.loc[(data['diag_3'] >= 390) & (data['diag_3'] < 460), 'diag_3'] = 7
    data.loc[(data['diag_3'] >= 460) & (data['diag_3'] < 520), 'diag_3'] = 8
    data.loc[(data['diag_3'] >= 520) & (data['diag_3'] < 580), 'diag_3'] = 9
    data.loc[(data['diag_3'] >= 580) & (data['diag_3'] < 630), 'diag_3'] = 10
    data.loc[(data['diag_3'] >= 630) & (data['diag_3'] < 680), 'diag_3'] = 11
    data.loc[(data['diag_3'] >= 680) & (data['diag_3'] < 710), 'diag_3'] = 12
    data.loc[(data['diag_3'] >= 710) & (data['diag_3'] < 740), 'diag_3'] = 13
    data.loc[(data['diag_3'] >= 740) & (data['diag_3'] < 760), 'diag_3'] = 14
    data.loc[(data['diag_3'] >= 760) & (data['diag_3'] < 780), 'diag_3'] = 15
    data.loc[(data['diag_3'] >= 780) & (data['diag_3'] < 800), 'diag_3'] = 16
    data.loc[(data['diag_3'] >= 800) & (data['diag_3'] < 1000), 'diag_3'] = 17
    data.loc[(data['diag_3'] == -1), 'diag_3'] = 0

    data['race'] = data['race'].replace('?', 'Other')

    # Here we have just 3 values that are Invalid, as a result we will change them into Female, which is the mod for this feature.
    data['gender'] = data['gender'].replace('Unknown/Invalid', 'Female')

    # Now we can easliy change Male/Female into 1/0
    data['gender'] = data['gender'].replace('Male', 1)
    data['gender'] = data['gender'].replace('Female', 0)

    # For age, we have 10 categories, each represents 10 years range from [0-10] to [90-100]. We will replace those with the middle age for each age range: for example (0,10] will be repleased with 5; (60, 70] will be replaces by 65; and so on.
    for i in range(0, 10):
        data['age'] = data['age'].replace(
            '['+str(10*i)+'-'+str(10*(i+1))+')', i*10+5)

    # Replace by 4 numerical categories
    data['max_glu_serum'] = data['max_glu_serum'].replace("None", 0)
    data['max_glu_serum'] = data['max_glu_serum'].replace("Norm", 1)
    data['max_glu_serum'] = data['max_glu_serum'].replace(">200", 2)
    data['max_glu_serum'] = data['max_glu_serum'].replace(">300", 3)

    # Replace by 4 numerical categories
    data['A1Cresult'] = data['A1Cresult'].replace("None", 0)
    data['A1Cresult'] = data['A1Cresult'].replace("Norm", 1)
    data['A1Cresult'] = data['A1Cresult'].replace(">7", 2)
    data['A1Cresult'] = data['A1Cresult'].replace(">8", 3)

    # All values in :    metformin,     glimepiride,    glipizide,    glyburide,    pioglitazone,    rosiglitazone,    insulin
    # can be No, Dowm, Steady and Up ...  replace them by 4 values
    drug_list = ['metformin', 'glimepiride', 'glipizide',
                 'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin']
    for i in drug_list:
        data[i] = data[i].replace('No', 0)
        data[i] = data[i].replace('Steady', 2)
        data[i] = data[i].replace('Down', 1)
        data[i] = data[i].replace('Up', 3)

    # Convert change into binary representation
    data['change'] = data['change'].replace('No', 0)
    data['change'] = data['change'].replace('Ch', 1)

    # Convert diabetesMed into binary representation
    data['diabetesMed'] = data['diabetesMed'].replace('Yes', 1)
    data['diabetesMed'] = data['diabetesMed'].replace('No', 0)

    # Race into binary categories
    data = pd.concat([data, pd.get_dummies(
        data['race'], prefix='race')], axis=1).drop(['race'], axis=1)

    # Convert readmitted into numerical representation
    if binary:
        data['readmitted'] = data['readmitted'].replace('NO', 0)
        data['readmitted'] = data['readmitted'].replace('>30', 1)
        data['readmitted'] = data['readmitted'].replace('<30', 1)
    else:
        data['readmitted'] = data['readmitted'].replace('NO', 0)
        data['readmitted'] = data['readmitted'].replace('>30', 1)
        data['readmitted'] = data['readmitted'].replace('<30', 2)

    # Divide labels and features
    y = data['readmitted']
    X = data.drop(['readmitted'], axis=1)

    return X, y


def statistical_parity(y, y_, Z, priv=None):
    if priv is None:
        values = np.unique(Z)
        counts = [np.mean(y[Z == z]) for z in values]
        priv = values[np.argmax(counts)]
        unpriv = [z for z in values if z != priv]
        print('Automatic priviledged value is', priv)
    else:
        unpriv = [z for z in values if z != priv]

    return np.array([np.mean([y_i for y_i, zi in zip(y_, Z) if zi == unp]) - np.mean([y_i for y_i, zi in zip(y_, Z) if zi == priv])
                     for unp in unpriv])


def average_odds(y, y_, Z, priv=None):
    if priv is None:
        values = np.unique(Z)
        counts = [np.mean(y[Z == z]) for z in values]
        priv = values[np.argmax(counts)]
        unpriv = [z for z in values if z != priv]
        print('Automatic priviledged value is', priv)
    else:
        unpriv = [z for z in values if z != priv]

    return np.array([1/2*(np.mean([y_i for y_i, yi, zi in zip(y_, y, Z) if zi == unp and yi == 1]) -
                          np.mean([y_i for y_i, yi, zi in zip(y_, y, Z) if zi == priv and yi == 1])) +
                     1/2*(np.mean([y_i for y_i, yi, zi in zip(y_, y, Z) if zi == unp and yi == 0]) -
                          np.mean([y_i for y_i, yi, zi in zip(y_, y, Z) if zi == priv and yi == 0]))
                     for unp in unpriv])


def average_predictive_value(y, y_, Z, priv=None):
    if priv is None:
        values = np.unique(Z)
        counts = [np.mean(y[Z == z]) for z in values]
        priv = values[np.argmax(counts)]
        unpriv = [z for z in values if z != priv]
        print('Automatic priviledged value is', priv)
    else:
        unpriv = [z for z in values if z != priv]

    return np.array([1/2*(np.mean([yi for y_i, yi, zi in zip(y_, y, Z) if zi == unp and y_i == 1]) -
                          np.mean([yi for y_i, yi, zi in zip(y_, y, Z) if zi == priv and y_i == 1])) +
                     1/2*(np.mean([yi for y_i, yi, zi in zip(y_, y, Z) if zi == unp and y_i == 0]) -
                          np.mean([yi for y_i, yi, zi in zip(y_, y, Z) if zi == priv and y_i == 0]))
                     for unp in unpriv])


def consistency(X, y_, k, distance=lambda x: np.linalg.norm(x, 1)):
    D_matrix = np.array([[distance(xi-xj) for xj in X] for xi in X])
    N = np.argsort(D_matrix+np.eye(D_matrix.shape[0])*10**10, axis=0)[:, :k]
    i_consist = [abs(y_[i]-np.mean([y_[N[i, j]] for j in range(k)]))
                 for i in range(y_.shape[0])]
    return 1 - np.mean(i_consist)


def theil_index(y, y_):
    b = (1-y+y_)/2
    b_ = np.mean(b)
    return np.mean(b/b_*np.log(b/b_+10**-10))


class CounterfactualPreProcessing(MetaEstimatorMixin, BaseEstimator):
    def __init__(self, estimator, sensitive_feature_ids):
        self.estimator = estimator
        self.sensitive_feature_ids = sensitive_feature_ids

    def fit(self, X, y):
        sensitive_columns = X[:, self.sensitive_feature_ids]
        self.unique_values = []
        for scol in sensitive_columns.T:
            self.unique_values += [np.unique(scol).tolist()]

        X_rows = []
        y_rows = []

        for xi, yi in zip(X, y):
            for comb in itertools.product(*self.unique_values):
                comb = np.array(comb)
                xi[self.sensitive_feature_ids] = comb
                X_rows += [xi.copy()]
                y_rows += [yi]

        X_transf = np.array(X_rows)
        y_transf = np.array(y_rows)

        print(X_transf.shape, y_transf.shape)

        self.estimator.fit(X_transf, y_transf)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


class Weighted(MetaEstimatorMixin, BaseEstimator):
    def __init__(self, estimator, sensitive_feature_ids):
        self.estimator = estimator
        self.sensitive_feature_ids = sensitive_feature_ids

    def fit(self, X, y):
        sensitive_columns = X[:, self.sensitive_feature_ids]
        sensitive_columns_y = np.append(
            X[:, self.sensitive_feature_ids], y[:, np.newaxis], axis=1)

        unique_feats = []
        for scol in sensitive_columns.T:
            unique_feats += [np.unique(scol).tolist()]
        unique_y = np.unique(y).tolist()

        unique_feats_y = []
        for scol in sensitive_columns_y.T:
            unique_feats_y += [np.unique(scol).tolist()]

        self.unique_values = []
        for scol in sensitive_columns.T:
            self.unique_values += [np.unique(scol).tolist()]
        self.unique_values += [np.unique(y).tolist()]

        sample_weight = np.ones(sensitive_columns_y.shape[0])
        for comb in itertools.product(*unique_feats):
            comb = np.array(comb)
            where = np.prod(sensitive_columns == comb, axis=1) != 0
            sample_weight[where] *= np.mean(
                np.prod(sensitive_columns == comb, axis=1))

        for comb in itertools.product(*[unique_y]):
            comb = np.array(comb)
            where = (y == comb[0])
            sample_weight[where] *= np.mean(y == comb[0])

        for comb in itertools.product(*unique_feats_y):
            comb = np.array(comb)
            where = np.prod(sensitive_columns_y == comb, axis=1) != 0
            sample_weight[where] /= np.mean(
                np.prod(sensitive_columns_y == comb, axis=1))

        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
