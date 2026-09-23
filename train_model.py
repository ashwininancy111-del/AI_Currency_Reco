import cv2
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

DATASET = "dataset"

images = []
labels = []

# Read images from each currency folder
for label in os.listdir(DATASET):

    folder = os.path.join(DATASET, label)

    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.resize(img, (64, 64))
        img = img.flatten()

        images.append(img)
        labels.append(label)

X = np.array(images)
y = np.array(labels)

print("Total images:", len(X))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train AI model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Test accuracy
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")