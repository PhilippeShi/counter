import cv2
import numpy as np
import PIL.Image
from sklearn.svm import LinearSVC 
import pickle
import os

class Model:

    def __init__(self):
        self.model = LinearSVC()

    def train_model(self, name=None):
        img_list = np.array([])
        class_list = np.array([])

        _, _, files = next(os.walk("1"))
        file_count1 = len(files)
        _, _, files = next(os.walk("2"))
        file_count2 = len(files)

        for i in range(1, file_count1):
            img = cv2.imread(f"1/frame{i}.jpg")[:,:,0]
            img = img.reshape(16950)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 1)
        
        for i in range(1, file_count2):
            img = cv2.imread(f"2/frame{i}.jpg")[:,:,0]
            img = img.reshape(16950)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 2)

        img_list = img_list.reshape(file_count1 - 1 + file_count2 - 1, 16950)
        self.model.fit(img_list, class_list)
        if not os.path.exists("models"):
            os.mkdir("models")
        pickle.dump(self.model, open(f'models/{name}.sav', 'wb'))
        print("Model successfully trained!")

    def predict(self, frame):
        frame = frame[1]
        cv2.imwrite("frame.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        img = PIL.Image.open("frame.jpg")
        img.thumbnail((150, 150), PIL.Image.ANTIALIAS)
        img.save("frame.jpg")

        img = cv2.imread("frame.jpg")[:,:,0]
        img = img.reshape(16950)
        prediction = self.model.predict([img])
        os.remove("frame.jpg")
        return prediction[0]

    def get_model(self, name):
        self.model = pickle.load(open(f'models/{name}.sav', 'rb'))