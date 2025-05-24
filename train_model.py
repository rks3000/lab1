
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)

# Train model
clf = LogisticRegression(max_iter=200)
clf.fit(X, y)

print("Model trained successfully!")


