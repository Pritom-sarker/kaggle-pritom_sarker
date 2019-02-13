import My_ML_Lib as my
import pandas as pd
import matplotlib.pyplot as plt

# predict-------------------------
x_train = my.get_data("test.csv", "test", "all", "")
X = my.prepare_data(x_train, "Id")
inn = [12, 13, 8, 10, 26, 24, 15, 2, 1, 7, 28, 11, 27, 5, 6]
x_final = my.drop_index(X, inn)
x_final["Id"]=x_train["Id"].tolist()
mod = my.load_model("finalized_model_using_regression_1.pkl")
my.save_ans(mod, x_final, "answer.csv", "Id", "SalePrice")
