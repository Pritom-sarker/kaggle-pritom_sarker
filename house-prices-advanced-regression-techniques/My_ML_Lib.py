import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle


def get_data(place,method,attr,terget):
    # Attribrute -> list of index we need (X)
    #terget -> single index (Y)

    if method=="train":
        train = pd.read_csv(place)
        # print(train.head())
        #numeric = ['Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        if attr =="all":
            x_train = train.drop(terget,1)
        else:
            x_train = train[attr]
        y_train = train[terget]
        return train,x_train,y_train

    elif method=="test":

        train = pd.read_csv(place)
        if attr =="all":
            x_train = train
        else:
            x_train = train[attr]

        return x_train

def plot_for_linear_relation(train,Y,output,x):
    import seaborn as sns
    import os
    # train -> Full dataset
    #Output -> terget data name
    #x -> Type 'show' if u want to see plot
    train[output]=Y.tolist()
    col=train.columns.values.tolist()
    inn=0
    for i in col:

        sns.pairplot(train, x_vars=[i], y_vars=output, size=7, aspect=0.7)


        plt.title("index ->{}".format(inn))
        inn += 1

        plt.savefig("./img/{}.jpg".format(i))
        if x=='show':
            plt.show()

def prepare_data(x_train,indexx):
    #Convert numeric data
    #indexx-- >> ID primary key of the table
    text=[]
    text_= x_train.select_dtypes(include="object")
    text=text_.columns.values.tolist()
    id=x_train[indexx].tolist()
    x_train.drop(indexx,1)
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    for i in text:
        housing_cat = x_train[i]
        housing_cat_encoded = encoder.fit_transform(housing_cat.astype(str))
        x_train = x_train.drop(i, 1)
        x_train[i] = housing_cat_encoded

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
    train_x[indexx]=id


    return train_x

def save_ans(model,test_x,file,indexx,ans):
    x=test_x.drop(indexx,1)
    pred = model.predict(x)
    # print(pred)

    data = pd.DataFrame({indexx: test_x[indexx], ans: pred})
    data.set_index(indexx, inplace=True)
    data.to_csv(file)
    print("Data saved [{}]!!".format(file))

def cross_validation(X,Y,model,fold):
    from sklearn.model_selection import cross_val_score

    forest_scores = cross_val_score(model, X, Y, cv=fold,
                                    scoring="neg_mean_squared_error")
    forest_rmse_scores = np.sqrt(-forest_scores)
    return forest_rmse_scores

def save_model(File_name,model):
    # file_name='finalized_model.pkl'

    from sklearn.externals import joblib
    joblib.dump(model, File_name)
    print("Model saved at {}".format(File_name))

def load_model(File_name):
    from sklearn.externals import joblib
    loaded_model = joblib.load(File_name)
    return loaded_model

def drop_index(X,index):
    col = X.columns.values.tolist()
    x_new=pd.DataFrame()
    for i in range(0,len(col)):

        if i not in index:
            X=X.drop(col[i],1)
    return X