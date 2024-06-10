from deep_learning.signClassifier import SignClassifier
from deep_learning.config import *
from utils.utils import *

clf = SignClassifier()
clf.fit()

X_val, y_val = load_dataset(VAL_IMAGE_FILE_PATH, VAL_LABEL_FILE_PATH)

y = clf.predict(X_val)

print("Le taux d'erreur est de ",np.mean(y!=y_val))
print(y, y_val)