import  pandas as pd
import My_ML_Lib as mml

if __name__ == '__main__':
    # data handle -----------------

    data=pd.read_csv('spam.csv' , encoding='latin-1')
    #print(data.head())

    x=data['v2']
    y=data['v1']


    #print(y.shape)



    # char input to str

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    Y=encoder.fit_transform(y.astype(str))

    from sklearn.feature_extraction.text import TfidfVectorizer

    cv = TfidfVectorizer(min_df=1,stop_words='english')
    X=cv.fit_transform(x).toarray()
    print(X.shape)


    # Feature selection

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2

    X_new = SelectKBest(chi2, k=2).fit_transform(X, Y)


    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    param_grid = {
        'C': [11],

    }
    from sklearn.svm import LinearSVC

    lin = LinearSVC(loss='hinge')
    grid_search = GridSearchCV(estimator=lin, param_grid=param_grid,
                               cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X, Y)
    print('Final Loss',grid_search.best_params_,"Final accuracy :", grid_search.best_score_*100,"%")

    from sklearn.metrics import accuracy_score

    best_grid = grid_search.best_estimator_
    #mml.cross_validation(X, Y, best_grid, 3)



    #For input

    xx=['''sir please call me''']
    X_data = cv.transform(xx)

    print(best_grid.predict(X_data))