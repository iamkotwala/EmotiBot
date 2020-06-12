            
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
import pylab as plt 
import pickle
import numpy as np

data = pickle.loads(open("tensorflow_prediction.pickle", "rb").read())
data = np.array(data)
actual = data[0]
predicted = data[1]
print(actual)
print(predicted)
print(len(actual))
print(len(predicted))
print("Accuracy Score :",accuracy_score(actual, predicted))
print("Report         :\n",classification_report(actual, predicted))

classes=[
	{
		"label":"angry",
		"train_location":"data/train/angry/*",
		"test_location":"data/test/angry/*"
	},
	{
		"label":"disgusted",
		"train_location":r"data\train\disgusted\*",
		"test_location":r"data\test\disgusted\*"
	},
    {
		"label":"fearful",
		"train_location":r"data\train\fearful\*",
		"test_location":r"data\test\fearful\*"
	},
	{
		"label":"happy",
		"train_location":r"data\train\happy\*",
		"test_location":r"data\test\happy\*"
	},
    {
		"label":"neutral",
		"train_location":r"data\train\neutral\*",
		"test_location":r"data\test\neutral\*"
	},
	{
		"label":"sad",
		"train_location":r"data\train\sad\*",
		"test_location":r"data\test\sad\*"
	},
    {
		"label":"surprised",
		"train_location":r"data\train\surprised\*",
		"test_location":r"data\test\surprised\*"
	}
]

labels = [d['label'].title() for d in classes]
print(labels)
cm = confusion_matrix(actual, predicted, labels)
fig, ax = plt.subplots(figsize=(12, 12))
cax = ax.matshow(cm,cmap=plt.cm.RdYlGn)
plt.title("3D Convolution Neural Network Confusion Matrix")
fig.colorbar(cax)
ax.grid(False)
predicted_text = []
actual_text = []
for clss in classes:
  predicted_text.append("Pred "+clss["label"])
  actual_text.append("Actual "+clss["label"])
ax.xaxis.set(ticks=(0, 1, 2, 3, 4, 5, 6), ticklabels=tuple(predicted_text))
ax.yaxis.set(ticks=(0, 1, 2, 3, 4, 5, 6), ticklabels=tuple(actual_text))
#ax.set_ylim(1.5, -0.5)
for i in range(7):
    for j in range(7):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.savefig('tensorflow_confusion.png')
print("Tensorflow Confusion Matrix has been saved to ./docs/tensorflow_confusion.png")
plt.show()