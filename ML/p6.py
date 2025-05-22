import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
data=sns.load_dataset("tips")
X= StandardScaler().fit_transform(data["total_bill"].values.reshape(-1,1))
y=data["tip"].values
def lwr(X_train,y_train,X_test,tau):
    y_pred=[np.r_[1,x] @ np.linalg.pinv(np.c_[np.ones_like(X_train),X_train].T @ np.diag(np.exp(-np.sum((X_train -x)**2, axis=1)/(2 *tau **2))) @ np.c_[np.ones_like(X_train),X_train])@(np.c_[np.ones_like(X_train),X_train].T @ np.diag(np.exp(-np.sum((X_train - x) **2,axis=1) / (2 * tau **2))) @ y_train) for x in X_test]
    return np.array(y_pred)
tau=0.5
X_test=np.linspace(X.min(),X.max(),100).reshape(-1,1)
y_pred=lwr(X,y,X_test,tau)
plt.scatter(X,y,color="blue",alpha=0.6,label="Data points")
plt.plot(X_test,y_pred,color="red",linewidth=2,label="LWR Fit")
plt.xlabel("Ttal bill (Standardized)")
plt.ylabel("Tip")
plt.title("Locally Weighted Regression")
plt.legend()
plt.show()
