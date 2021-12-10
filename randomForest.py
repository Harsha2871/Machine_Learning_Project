from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier





def randomForest(X,Y):
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    scores = cross_val_score(forest, X, Y, cv=5)
    return sum(scores)/len(scores)


