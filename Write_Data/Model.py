from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class model:
    def __init__(self, target, features):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.target = target
        self.features = features
        self.X_train = 0
        self.X_test = 0
        self.y_train = 0
        self.y_test = 0

    def model_train(self):
        X_train, X_test, y_train , y_test = train_test_split(self.features, self.target, test_size= 0.2 , random_state= 42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def model_fit(self):
       self.model.fit(self.X_train, self.y_train)

    def model_predict(self):
        predictions = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, predictions)
