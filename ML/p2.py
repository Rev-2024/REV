import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('housing.csv')
data_numeric = data.select_dtypes(include=[float,int])
correlation_matrix = data_numeric.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt='.2f',linewidth=0.5)
plt.show()
sns.pairplot(data_numeric,diag_kind='kde',plot_kws={'alpha':0.5})
plt.suptitle('pair plot of california Housin features',y=1.02)
plt.show()
