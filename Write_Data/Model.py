from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class model:
    def __init__(self, target, features):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.target = target
        self.features = features
        



    def model_fit(self):
       self.model.fit(self.features, self.target)

    def model_predict(self):
        predictions = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, predictions)

    def forrecast(self, MW ,probability):
        Forecasted_Load = MW * probability
        return Forecasted_Load
        #Forecasted_Load = Requested MW * Probability of Realization
