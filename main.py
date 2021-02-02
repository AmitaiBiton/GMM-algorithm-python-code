import numpy as np
from sklearn.datasets import load_digits, make_friedman1
import matplotlib.pyplot as plt
import  random
from scipy.stats import multivariate_normal
import pandas as pd
from sklearn import metrics
import seaborn as sn
def randomClasses():
    stop =0
    while stop ==0:
        c1 = random.randint(0, 9)
        c2 =random.randint(0, 9)
        c3 =random.randint(0, 9)
        if c1!=c2 and c1!=c3 and c2!=c3 :
            stop =1
    return c1, c2 ,c3


def getDataFor3Classes(X ,Y , c1 ,c2 ,c3) :
    newX =[]
    newY = []
    for i in range(X.shape[0]):
        if Y[i]==c1 or Y[i]==c2 or Y[i]==c3:
            newX.append(X[i])
            newY.append(Y[i])
    return newX , newY

def initialization(X , numClasses):

    pi = np.ones(numClasses)/numClasses
    mu = np.random.randint(min(X[:, 0]), max(X[:, 0]), size=(numClasses, len(X[0])))
    cov = np.zeros((numClasses, len(X[0]), len(X[0])))

    for n in range(len(cov)):
        np.fill_diagonal(cov[n], 5)
    return pi , mu , cov


def GMM_RUN(X , Y , mu , cov , pi , numOfIterions):
    reg_cov = 1e-6 * np.identity(len(X[0]))
    x, y = np.meshgrid(np.sort(X[:, 0]), np.sort(X[:, 1]))

    # -------------E step-------------------
    for  iters in range(numOfIterions):
        # wil contain for any example array with size 3 for calculate
        # for class 1 thr Pr that this point belongs to this class...
        ric = np.zeros((len(X), len(mu)))
        for pic, muc, covc, r in zip(pi, mu, cov, range(len(ric[0]))):
            covc += reg_cov
            mn = multivariate_normal(mean=muc, cov=covc)
            # pr(Xi | Zi=j) * pr(Zi = j)
            ric[:, r] = pic * mn.pdf(X)

        # ric = ric / pr(Xi)
        for r in range(len(ric)):
            ric[r, :] = ric[r, :] / np.sum(ric[r, :])

        #---------- step M --------------

        # update => pi = 1/m*Wi,j
        #  sum the Wi,j that for class 1 all the pr equal to some number
        mc = np.sum(ric, axis=0)

        # pi = 1/m * mc and  pi[0] + pi[1] + pi[2] ====1
        pi = mc / np.sum(mc)

        # update mu by dot between ric.T to  X and  / sum of ric
        # for two feature abd 3 classes 2X3 matrix
        mu = np.dot(ric.T, X)/ mc.reshape(3,1)

        cov = []
        # update cov
        for r in range(len(pi)):
            # Wj,i * (Xi-MUj )*(Xi-MUj).T
            covc = 1/ mc[r] * (
                        np.dot((ric[:, r].reshape(len(X), 1) * (X - mu[r])).T, X - mu[r]) + reg_cov)
            cov.append(covc)
        # list to array
        cov = np.asarray(cov)

    return  pi , mu , cov

def predicted(X , pi , mu , cov):
    # will contain matrix 3X example dim
    # and for any row we take the class that have max on P

    predictions = []
    reg_cov = 1e-6 * np.identity(len(X[0]))
    for pic, m, c in zip(pi, mu, cov):
        c += reg_cov
        mn = multivariate_normal(mean=m, cov=c)
        prob = mn.pdf(X)
        predictions.append([prob])

    predictions = np.asarray(predictions).reshape(len(X),3)
    # max on any Wi,j to get the class is belongs
    predictions = np.argmax(predictions, axis=1)
    return  predictions


def change_ypred_number_to_Y(newY , y_pred):
    y_copy = np.copy(newY)
    new_y_pred = np.copy(y_pred)

    y  = np.where(y_copy==np.amin(y_copy))
    y1 = y_copy[y][0]
    new_y_pred[y_pred==0]=y1
    y_copy[y] =300

    y = np.where(y_copy == np.amin(y_copy))
    y2 = y_copy[y][0]
    new_y_pred[y_pred==1]=y2
    y_copy[y] = 300

    y = np.where(y_copy == np.amin(y_copy))
    y3 = y_copy[y][0]
    new_y_pred[y_pred==2]=y3

    return new_y_pred






if  __name__ == "__main__" :
    X_digits, y_digits = load_digits(return_X_y=True)

    #X = StandardScaler().fit_transform(X_digits)

    # get the best two feature in X
    matrix_X = X_digits[: , (28,61)]

    # chose three classes 0-9
    c1 , c2 , c3  = randomClasses()

    newX  , newY= getDataFor3Classes(matrix_X ,y_digits, c1, c2 ,c3)
    newX = np.array(newX)
    newY = np.array(newY)
    numClasses = 3
    pi  ,mu , cov= initialization(newX , numClasses)

    pi , mu , cov = GMM_RUN(newX,newY, mu, cov, pi , numOfIterions=100)
    predictions = predicted(newX,  pi , mu , cov)
    predictions = change_ypred_number_to_Y(newY ,predictions)
    Accuracy = metrics.accuracy_score(newY, predictions)
    print('Accuracy: ', Accuracy)
    cm = pd.crosstab(newY.flatten(), predictions, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(cm, annot=True)
    plt.title('n= '+str(newX.shape[0])+', Accuracy = '+str(Accuracy) )
    plt.show()



