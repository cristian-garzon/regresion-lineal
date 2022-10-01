import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model, model_selection

matplotlib.use('TkAgg')


def regLineal():
    df = pd.read_csv("./regresionSimple/data.csv")
    df = pd.DataFrame(df)
    x = df[["edad"]]
    y = df[["inseguridad"]]
    plt.scatter(x, y)
    plt.xlabel("edad")
    plt.ylabel("inseguridad")

    plt.show()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.20)
    regresion = linear_model.LinearRegression()
    regresion.fit(x_train, y_train)

    Y_pred = regresion.predict(x_test)
    plt.scatter(x_train, y_train)
    plt.plot(x_test, Y_pred, color="red", linewidth=3)
    plt.title("regresion lineal")
    print(f"la regresion es de: {regresion.score(x, y)}")

    plt.show()
    print(f"y {regresion.coef_}X + {regresion.intercept_}")

