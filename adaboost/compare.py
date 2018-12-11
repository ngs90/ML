from model import AdaBoost, Loss, test_data
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import time

if __name__ == '__main__':
    y, X, w, x1_0, x2_0, x1_1, x2_1 = test_data()
    #y = y[400:600]
    #X = X[400:600]


    time_it = time.time()
    adab = AdaBoost(binary_search=True)
    adab.train_model(X=X, y=y, trees=1, depth=5, w=None)
    pred_proba = adab.predict(proba=True)
    pred = adab.predict(proba=False)
    print('time taken:', time.time()-time_it)
    print('Weighted loss:', Loss(y=y, y_p=pred, metric='weighted_error', w=None).loss_value)
    print('Misclassification error rate:', Loss(y=y, y_p=pred, metric='misclassification_error', w=None).loss_value)
    print('Accuracy:', accuracy_score(y_true=y, y_pred=pred))

    time_it = time.time()
    adab_sk = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=1)
    adab_sk.fit(X=X, y=y)
    pred = adab_sk.predict(X=X)
    pred_proba = adab_sk.predict_proba(X=X)
    print('time taken:', time.time()-time_it)
    print('Weighted loss:', Loss(y=y, y_p=pred, metric='weighted_error', w=None).loss_value)
    print('Misclassification error rate:', Loss(y=y, y_p=pred, metric='misclassification_error', w=None).loss_value)
    print('Accuracy:', accuracy_score(y_true=y, y_pred=pred))

