import cv2
import numpy as np
import sklearn as sk
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from scipy.stats import sem

PATH_TO_TRAINING_DATA = '../training_data'


def print_faces(images, top_n):
    fig = plt.figure(figsize=(12,12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(top_n):
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
    plt.show()

def evaluate_cross_validation(clf, X, y, K):
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print("Mean score: {0:.3f} (+/- {1:.3f})".format(np.mean(scores), sem(scores)))

if __name__ == "__main__":
    #   img = cv2.imread(PATH_TO_TRAINING_DATA + '/barack_obama.jpg', 0)
    faces = fetch_olivetti_faces()
    # print_faces(faces.images, 20)
    svc_1 = SVC(kernel='linear')

    #split dataset into training and test datasets
    X_train, X_test, Y_train, Y_test = train_test_split(faces.data, faces.target, test_size=0.25, train_size=0.75, random_state=0)

    evaluate_cross_validation(svc_1, X_train, Y_train, 5)

