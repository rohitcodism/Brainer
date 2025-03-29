import os
import cv2
import matplotlib.pyplot as plt

def plot_images(folder, n=5):
    images = os.listdir(folder)[:n]

    plt.figure(figsize=(15,5))

    for i, img_name in enumerate(images):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.title(img_name)
        plt.axis('off')
    plt.show()

def load_images(folder, label, img_size=(224,224)):
    images=[]
    labels=[]

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    return images, labels
