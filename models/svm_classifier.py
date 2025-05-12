from sklearn import svm
import joblib

def train_svm(X, y):
    clf = svm.SVC(kernel='rbf', probability=True)
    clf.fit(X, y)
    joblib.dump(clf, 'models/svm_model.pkl')

def predict_svm(features):
    clf = joblib.load('models/svm_model.pkl')
    return clf.predict(features), clf.predict_proba(features)