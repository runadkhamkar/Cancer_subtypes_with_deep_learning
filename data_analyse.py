
# In[]
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
path="./results/11.5.21/"
l=os.listdir(path)
df=[]
for i in l:
    if(i[-3:]=='tsv'):
        f=pd.read_csv(path+i,sep='\t')
        df.append(f)
x=['DT','GuassianNB','GBC','KNN','LDA','Logisticreg','MLP','QDA','RF','SVC','VotingC','XGB']
data=[[] for i in range(3)]
y=['FineTunning','UnSupervised','LineraSVC','PCA']
k=0
L=dict()
st=['FineTunned','Unsupervised','linearSVC','PCA']
for i in df:
    L[st[k]+'1']=x
    L[st[k]]=list(i['Recall'])
    k+=1
ndf=pd.DataFrame(L)

plt.plot('FineTunned1','FineTunned',data=ndf,marker='o',linewidth=1)
plt.plot('Unsupervised1','Unsupervised',data=ndf,marker='o',linewidth=1)
plt.plot('linearSVC1','linearSVC',data=ndf,marker='o',linewidth=1)
plt.plot('PCA1','PCA',data=ndf,marker='o',linewidth=1)
plt.legend()
plt.xlabel("Classifiers")
plt.ylabel("Accuracy")
plt.title('UCEC Dataset')
plt.show()
 

# In[]
