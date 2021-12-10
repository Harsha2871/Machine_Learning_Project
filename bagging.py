from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier

def bagging(X,Y):
    model = BaggingClassifier()
    scores = cross_val_score(model, X, Y, cv=5)
    return sum(scores)/len(scores)
