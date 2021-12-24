import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

y_pred=["ant", "ant", "cat", "cat", "ant", "cat"]  
y_true=["cat", "ant", "cat", "cat", "ant", "bird"] 
C = metrics.confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
plt.matshow(C, cmap=plt.cm.Greens)
plt.colorbar()
for i in range(len(C)):
    for j in range(len(C)):
        plt.annotate(C[i,j], xy=(i, j), horizontalalignment='center', verticalalignment='center')

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
