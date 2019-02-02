#Read data
import pandas as pd
import matplotlib.pyplot as plt

def get_data(place,method):
    if method=="train":
        train = pd.read_csv(place)
        # print(train.head())
        numeric = ['Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        x_train = train[numeric]
        y_train = train['Survived']
        return train,x_train,y_train
    elif method=="test":

        train = pd.read_csv(place)
        numeric = ['PassengerId','Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        x_train = train[numeric]
        return x_train

def prepare_data(x_train,method):
    #Convert numeric data
    if method=="test":
        id=x_train['PassengerId'].tolist()
        x_train.drop('PassengerId',1)
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    housing_cat = x_train["Sex"]
    housing_cat_encoded = encoder.fit_transform(housing_cat)
    x_train=x_train.drop("Sex",1)
    x_train["Sex"]=housing_cat_encoded

    housing = x_train['Embarked'].astype(str)
    housing = encoder.fit_transform(housing)
    x_train=x_train.drop('Embarked',1)
    x_train['Embarked']=housing


    #Create Pipeline
    from sklearn.preprocessing import Imputer as SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
        ])

    x_train_arry = num_pipeline.fit_transform(x_train)

    train_x=pd.DataFrame(x_train_arry,columns=x_train.columns)
    if method == "test":
        train_x['PassengerId']=id


    return train_x


def plot_graph(X):
    co_matrix=X.corr()
    print(co_matrix["Age"].sort_values(ascending=False))

    from pandas.plotting import scatter_matrix
    numeric = ['Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    scatter_matrix(X[numeric], figsize=(12, 8))
    plt.savefig("x.jpg")

    down = X['Sex'].tolist()
    avg = X['Age'].tolist()
    prob =X['Survived'].tolist()

    # plt.show()

    y = down
    z = avg
    n = prob

    fig, ax = plt.subplots()
    ax.scatter(y, z)
    plt.xlabel("Number Of Downloads", color="Green")
    plt.ylabel("Average Rating", color="Green")
    plt.title("High demand but less rating")

    for i, txt in enumerate(n):
        ax.annotate(txt, (y[i], z[i]))
    plt.savefig("plot-5.jpg")
    plt.show()


def model(X,Y):

    #using logistic regression
    from sklearn.linear_model import LogisticRegression
    log_reg=LogisticRegression()
    log_reg.fit(X,Y)

    #Using Gradient decent
    from sklearn.linear_model import SGDClassifier
    sgd=SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    sgd.fit(X,Y)

    return log_reg,sgd


def save_ans(data,test_x,file):
    x=test_x.drop("PassengerId",1)
    pred = data.predict(x)
    # print(pred)

    data = pd.DataFrame({"PassengerId": test_x["PassengerId"], "Survived": pred})
    data.set_index('PassengerId', inplace=True)
    data.to_csv(file)
    print("Data saved [{}]!!".format(file))

if __name__ == "__main__":
    train,X,Y=get_data("titanic/train.csv","train")
    X=prepare_data(X,"")
    #plot_graph(train)
    log,sgd=model(X,Y)
    test_x= get_data("titanic/test.csv","test")
    test = prepare_data(test_x,"test")
    save_ans(log,test,"answer_logistic.csv")
    save_ans(sgd,test,"answer_SGD.csv")
