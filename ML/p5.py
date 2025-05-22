import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
np.random.seed(42)
X=np.random.rand(100,1)
y=np.array(['class1' if x<=0.5 else 'class2' for x in X.flatten()])
X_train,X_test,y_train,y_test=X[:50],X[50:],y[:50],y[50:]
plt.figure(figsize=(12,8))
k_values=[1,2,3,4,5,20,30]
for i,k in enumerate(k_values,1):
    knn=KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    plt.subplot(3, 3, i)
    plt.scatter(X_test,y_test,color='red',label='True')
    plt.scatter(X_test,y_pred,color='yellow',marker='x',label='Predicted')
    plt.title(f"KNN k={k}")
    plt.xlabel("x label")
    plt.ylabel("class")
    plt.legend()
plt.tight_layout()
plt.show()
for k in k_values:
    knn=KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
    print(f"accuracy for k={k}:{knn.score(X_test,y_test):.2f}")

    
