import argparse as ap
import glob
import os

from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from config import *

if __name__ == '__main__':
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-p', '--pos_features_path', help='Path to the positive features directory',
                        required=True)
    args = vars(parser.parse_args())

    pos_feat_path = args['pos_features_path']


    print('Training a Linear SVM classifier:')
    X = []
    y = []

    print('Loading positive samples...')
    i = 0
    for feat_path in glob.glob(os.path.join(pos_feat_path, '*.feat')):
        if i == POS_SAMPLES:
            break

        x = joblib.load(feat_path)
        X.append(x)
        y.append(1)

        i += 1


    X_train = np.array(X)
    y_train = np.array(y)
    del X
    del y

    print('Training a Linear SVM Classifier...')
    clf = LinearSVC(random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    # If model directory doesn't exist, create one
    if not os.path.isdir(os.path.split(MODEL_PATH)[0]):
        os.makedirs(os.path.split(MODEL_PATH)[0])
    joblib.dump(clf, MODEL_PATH, compress=3)
    print('Classifier saved to {}'.format(MODEL_PATH))
