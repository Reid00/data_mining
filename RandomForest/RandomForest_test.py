import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_auc_score
from matplotlib.colors import ListedColormap

dataset=pd.read_csv(r'D:\python\GitRepository\100-Days-Of-ML-Code\datasets\Social_Network_Ads.csv')
print(dataset.head())
print(dataset.describe())
print(dataset.info())
print('==='*9)
print(dataset.isnull().sum())

x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

# 将数据集拆分成训练集和测试集
X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.25,random_state=0)

# 特征缩放
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test= sc.fit_transform(X_test)


# 调试训练集的随机森林
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

# 预测测试集结果
y_pred=classifier.predict(X_test)

# 生成混淆矩阵，以及准确率、召回率分析结果
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test,y_pred)
cr= classification_report(y_test,y_pred)

# 打印混淆矩阵 cm 以及分析结果（准确率，召回率，F1值）cr：
print(cm,'\n',cr)

# 计算出ROC AUC
roc_value = roc_auc_score(y_test, y_pred)
print(f'roc_value is: \n {roc_value}')

# 将训练集结果可视化
from matplotlib.colors import ListedColormap
X_set,y_set= X_train,y_train

X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max() + 1, step=0.01),
                    np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max() + 1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())


for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# 将测试集结果可视化
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

print(r'=============调参实例分界线=============')
# 调整参数 n_estimators
param_test1 = {'n_estimators': list(range(10,71,10))}
classifier = RandomForestClassifier(n_estimators=10, min_samples_split=20,min_samples_leaf=2,max_depth=9,criterion='entropy', oob_score=True,random_state=0)

gsearch1= GridSearchCV(estimator=classifier,param_grid=param_test1,scoring='roc_auc',cv=5)
gsearch1.fit(X_train,y_train)
print(gsearch1.cv_results_)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

# 调整参数 max_depth
