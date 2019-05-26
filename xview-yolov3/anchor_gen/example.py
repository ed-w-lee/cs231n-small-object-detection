import glob
import xml.etree.ElementTree as ET

import numpy as np
import json
from PIL import Image
from tqdm import trange

from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = '/home/derekhuang/gcloud/project/train_images/xView_train.geojson'
CLUSTERS = 30

def load_dataset(path):
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)
    dataset = []
    counter = 0
    for i in trange(len(data['features'])):
        if data['features'][i]['properties']['bounds_imcoords'] != []:
            image = data['features'][i]['properties']['image_id']
            #path = "/home/derekhuang/gcloud/project/train_images/{}".format(image)
            #print("Processing image: " + image)
            #width, height = Image.open(path).size
            width, height = 1, 1
            val = np.array([int(num) for num in data['features'][i]['properties']['bounds_imcoords'].split(",")])
            x = val[2]/width - val[0]/width
            y = val[3]/height - val[1]/height
            if x * y < 1024 and x * y > 0:
                dataset.append([x, y])
    print("Total small objects: " + str(len(dataset)))
    return np.array(dataset)


data = load_dataset(ANNOTATIONS_PATH)
print("Data loaded")
out = kmeans(data, k=CLUSTERS)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}".format(out))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))
