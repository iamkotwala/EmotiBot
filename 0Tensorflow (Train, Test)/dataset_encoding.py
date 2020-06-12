from skimage.transform import resize
import numpy as np
from skimage import io
import glob, os, pickle
from keras.utils import to_categorical
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


batch_size = 64
image_height = 48
image_width = 48
layers = 1
#Layers for RGB
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


sampleDataset = [[],[],[],[]]
states = ["train","test"]
for class_index, clss in enumerate(classes):
	for state_index,state in enumerate(states):
		files = glob.glob(clss[state+"_location"])
		print(clss["label"].title()+" "+state.title()+" Size :",len(files))

		for file_index in range(len(files)):
			image = io.imread(files[file_index])
			
			if len(image.shape) == 2:
				sampleDataset[(state_index*2)+0].append(resize(image, (image_height, image_width)))
				sampleDataset[(state_index*2)+1].append(class_index)
			
		print("")
	print("")
for index, sample in enumerate(sampleDataset):
	sampleDataset[index] = np.array(sample)
  
mean = np.array([0.5,0.5,0.5])
std = np.array([1,1,1])
sampleDataset[0] = sampleDataset[0].astype('float')
sampleDataset[2] = sampleDataset[2].astype('float')
temp = len(sampleDataset[0])
print(temp)
for i in range(layers):
	sampleDataset[0][:,:,i] = (sampleDataset[0][:,:,i]- mean[i]) / std[i]
	sampleDataset[2][:,:,i] = (sampleDataset[2][:,:,i]- mean[i]) / std[i]
num_iterations = int(len(sampleDataset[0])/batch_size) + 1
sampleDataset[1] = to_categorical(sampleDataset[1], len(classes))
sampleDataset[3] = to_categorical(sampleDataset[3], len(classes))

print("Saving Pickle Files")
for index, sample in enumerate(sampleDataset):
	print("sampleDataset["+str(index)+"] Length :"+str(len(sample)))
	with open("./sampleDataset"+str(index)+".pickle", "wb") as f:
		pickle.dump(len(sample), f)
		
		for data_index, data in enumerate(sample):
			
			pickle.dump(data, f)