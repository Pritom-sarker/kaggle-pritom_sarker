import My_ML_Lib as my
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    train, x_train, Y=my.get_data("train.csv","train","all","SalePrice")
    X=my.prepare_data(x_train,"Id")
    X=X.drop("Id",1)
  #  print(X.head())
    co_matrix = X.corr()
    x=""
    col=x_train.columns.values.tolist()
    for i in col:
        sns.pairplot(train, x_vars=[i], y_vars='SalePrice', size=7, aspect=0.7)
        plt.savefig("Linear_plot/{}.jpg".format(i))
        if x=='show':
            plt.show()

    from pandas.plotting import scatter_matrix

    # indx = X.columns.values.tolist()
    # scatter_matrix(X[indx[15:]], figsize=(120, 80))
    # plt.savefig("z2.jpg")
    # plt.show()
