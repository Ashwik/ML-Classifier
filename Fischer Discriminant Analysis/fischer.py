
###import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


### transform the points into one dimension

def fischerLDA(dataset ,pos):
    print(dataset)
    data = pd.read_csv(dataset,header=None)
    data_0 = data[data[pos]==0].iloc[:,0:-1].values
    data_1 = data[data[pos]==1].iloc[:,0:-1].values
    var1 = np.cov(np.transpose(data_0))
    var2 = np.cov(np.transpose(data_1))
    sw = var1 + var2
    sw_inv = np.linalg.inv(sw)
    
    mean1 = np.mean(data_0,axis=0)
    mean2 = np.mean(data_1,axis=0)
    
    w = np.dot(sw_inv,mean2-mean1)
    w_trans = np.transpose(w)
    val0= []
    val1 = []
    for i in range(0,len(data_0)):
        val0.append(np.dot(w_trans,data_0[i,:]))
                    
    for i in range(0,len(data_1)):
        val1.append(np.dot(w_trans,data_1[i,:])) 
        
    ###plot normal curves 
    
    fig,ax = plt.subplots();
    ax.set_xlim([-10,10])
    ax.set_ylim([0,0.6])  
    x_min,x_max = plt.xlim()
    
    m0 = np.mean(data_0)
    std0 = np.std(data_0)
    
    x = np.linspace(x_min,x_max,10000)    
    y_0 = norm.pdf(x,m0,std0)
    ax.plot(x,y_0,color='red',label='çlass0')
    
    m1 = np.mean(data_1)
    std1 = np.std(data_1)
    
      
    y_1 = norm.pdf(x,m1,std1)
    ax.plot(x,y_1,color='blue',label='class1')
    
    ##find the intersection point
    inter_p = 0;
    dif = y_0 - y_1
    for i in range(len(dif)-1):
        if(dif[i]==0):
            inter_p = x[i]
        elif(dif[i]*dif[i-1]<0):
            inter_p = x[i]
            
    print("Intersection point: ",inter_p)
    plt.title('Normal Distribution of the points of dataset'+str(pos-1))
    plt.show()
    
    
    ###plot the transformed points in oned dimension
    fig,axis = plt.subplots()
    axis.set_ylim([-2.5,2.5])
    axis.scatter(val0,[0]*len(val0),color='red')
    axis.scatter(val1,[0]*len(val1),color='blue')
    plt.title('One Dimensional Transformation of the  dataset'+str(pos-1))
    plt.show()
       
        
    ###accuracy and fscore
    TN = 0
    FN = 0
    TP = 0
    FP = 0
    
    for i in range(len(val0)):
        if(val0[i]<inter_p):
            TN+=1
        else:
            FN+=1
    
    for i in range(len(val1)):
        if(val1[i]>inter_p):
            TP+=1
        else:
            FP+=1
    
    Accuracy = (TP+TN)/float((TP+FP+FN+TN))
    print("Accuracy:",Accuracy)
    
    Precision = TP/float((TP+FP))
    Recall = TP/float((TP+FN))
    F_Score = 2*(Recall * Precision) / float((Recall + Precision))
    print("F-Score:",F_Score)


def main():
    fischerLDA("a1_d1.csv",2)
    fischerLDA("a1_d2.csv",3)

main()