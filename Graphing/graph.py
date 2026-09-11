import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class graph:
    def __init__(self, feature, target):
        self.df = pd.DataFrame({"Feature": feature, "Target": target})

    def graph(self):
        sns.lmplot(
        x="Feature",
        y="Target",
        data= self.df,
        logistic=True,
        y_jitter=0.03,
        line_kws={"color": "red"},
    )
    plt.title("Logistic Regression Curve (Seaborn)")
    plt.show()

