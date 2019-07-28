

__Name__ = ["Shah Soumil Nitin"]
__Email__ = ["soushah@my.bridgeport.edu","shahsoumil519@gmail.com"]
__Version__ = "1.0.0"
__Github__ = "https://github.com/soumilshah1995"
__Website__ = "https://soumilshah.herokuapp.com/"
__Blog__ = "https://soumilshah1995.blogspot.com/"
__Youtube__ = "https://www.youtube.com/channel/UC_eOodxvwS_H7x2uLQa-svw?view_as=subscriber"
__FaceBook__ = "https://www.facebook.com/soumilshah1995/"
__Project__ = "https://soumilshah.herokuapp.com/project"

Description = """
    Hello! I’m Soumil Nitin Shah, a Software and Hardware Developer based in New York City.
    I have completed by Bachelor in Electronic Engineering and my Double master’s in Computer and Electrical Engineering.
    I Develop Python Based Cross Platform Desktop Application , Webpages , Software, REST API, Database and much more
    I have more than 2 Years of Experience in Python
"""

try:

    import pandas as pd
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer

    # Reporting the Data
    from sklearn.metrics import classification_report, confusion_matrix
    import scikitplot as skplt


    # Linear Classification Models
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import learning_curve, GridSearchCV
    from sklearn.linear_model import SGDClassifier
except:
    print("Some Modules are Missings ...... ")


class SciKitMaster(object):

    def __init__(self):
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = self.preprocess_data

        self.param_grid = {'C': [0.1, 1, 10, 100, 1000],
                           'gamma': [1, 0.1, 0.01, 0.01, 0.0001]}
        self.Models = {
            "model_logistic_rg": LogisticRegression(),
            "model_multibinomial": MultinomialNB(),
            "model_svc": GridSearchCV(SVC(), param_grid=self.param_grid),
            'model_decision_tree': DecisionTreeClassifier(),
            'model_random_forest': RandomForestClassifier(),
            'model_SGD': SGDClassifier(loss="hinge", penalty="l2", max_iter=100)}

    @property
    def preprocess_data(self):

        df = pd.DataFrame(data=load_breast_cancer()["data"], columns=load_breast_cancer()["feature_names"])

        df["target"] = load_breast_cancer()["target"]

        X_Data = df[['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                     'mean smoothness', 'mean compactness', 'mean concavity',
                     'mean concave points', 'mean symmetry', 'mean fractal dimension',
                     'radius error', 'texture error', 'perimeter error', 'area error',
                     'smoothness error', 'compactness error', 'concavity error',
                     'concave points error', 'symmetry error', 'fractal dimension error',
                     'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                     'worst smoothness', 'worst compactness', 'worst concavity',
                     'worst concave points', 'worst symmetry', 'worst fractal dimension']]

        Y_Data = df["target"]
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Data, Y_Data, test_size=0.4, random_state=101)

        return X_Train, X_Test, Y_Train, Y_Test

    @property
    def Train_Test_Architecture(self):

        for counter, model in enumerate(self.Models):
            print("="*65)
            print("\n")
            print("\t{}\t{}".format(counter, model))
            print("\n")

            print(self.Models.get(model))
            print("\n")
            model = self.Models.get(model)
            model.fit(self.X_Train, self.Y_Train)
            pred = model.predict(self.X_Test)

            report = classification_report(self.Y_Test, pred)
            print(report)
            print("\n")
            print(confusion_matrix(self.Y_Test, pred))
            print("\n")
            skplt.metrics.plot_confusion_matrix(self.Y_Test, pred, figsize=(6,6), title="Confusion Matrix {}".format(counter))



if __name__  == "__main__":
    neural = SciKitMaster()
    neural.Train_Test_Architecture