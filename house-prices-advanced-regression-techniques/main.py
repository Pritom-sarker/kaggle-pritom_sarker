import My_ML_Lib as my
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":

    #-------------> Load data
    train, x_train, Y=my.get_data("train.csv","train","all","SalePrice")
    X=my.prepare_data(x_train,"Id")
    X=X.drop("Id",1)
    co_matrix = X.corr()

    #print(co_matrix['MSSubClass'].sort_values(ascending=False))
    #my.plot_for_linear_relation(X,Y,"SalePrice","")

    #-------------> check only linear attribrute

    inn=[12,13,8,10,26,24,15,2,1,7,28,11,27,5,6]
    x_final=my.drop_index(X,inn)
    print(x_final.head())

    #-------------> Train Model

    from sklearn.linear_model import SGDRegressor
    sgd_reg = SGDRegressor(max_iter=10000000, tol=-np.infty, penalty=None, eta0=0.1, random_state=42)
    sgd_reg.fit(x_final,Y)

    #-------------> Save Model

    my.save_model('finalized_model_using_regression_1.pkl', sgd_reg)

    #-------------> Cross validation
    print(my.cross_validation(x_final,Y,sgd_reg,3).mean())


