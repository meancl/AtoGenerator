from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours, OneSidedSelection, ClusterCentroids

def overSampleSMOTE(X, y, ratio=1.0):
    smote = SMOTE(sampling_strategy=ratio)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def overSampleRandom(X, y, ratio=1.0):
    randomOverSampler = RandomOverSampler(sampling_strategy=ratio)
    X_resampled, y_resampled = randomOverSampler.fit_resample(X, y)
    return X_resampled, y_resampled

def overSampleADASYN(X, y, ratio=1.0):
    adasyn = ADASYN(sampling_strategy=ratio)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    return X_resampled, y_resampled

def underSampleRandom(X, y, ratio=1.0):
    randomUnderSampler = RandomUnderSampler(sampling_strategy=ratio)
    X_resampled, y_resampled = randomUnderSampler.fit_resample(X, y)
    return X_resampled, y_resampled

def underSampleTomek(X, y, ratio=1.0):
    tomekLinks = TomekLinks(sampling_strategy=ratio)
    X_resampled, y_resampled = tomekLinks.fit_resample(X, y)
    return X_resampled, y_resampled

def underSampleEditedNN(X, y, ratio=1.0):
    editedNN = EditedNearestNeighbours(sampling_strategy=ratio)
    X_resampled, y_resampled = editedNN.fit_resample(X, y)
    return X_resampled, y_resampled

def underSampleOneSided(X, y, ratio=1.0):
    oneSided = OneSidedSelection(sampling_strategy=ratio)
    X_resampled, y_resampled = oneSided.fit_resample(X, y)
    return X_resampled, y_resampled

def underSampleClusterC(X, y, ratio=1.0):
    clusterC = ClusterCentroids(sampling_strategy=ratio)
    X_resampled, y_resampled = clusterC.fit_resample(X, y)
    return X_resampled, y_resampled
