from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
import numpy as np


def ModelTest():
    d = {}

    x2 = Input((1, 3), name="x2")
    x2 = Input((1, 3), name="x2")
    x1 = Input((1, 3), name="x1")

    d["b"] = x2
    d["b"] = x2
    d["a"] = x1
    print(x1, x2)
    input_list = list(d.values())
    x3 = Input((None, 4), name="x3")

    input_list.append(x3)
    print(input_list)
    y = Dense(1, activation='linear')(x3)
    model = Model(inputs=input_list, outputs=y)
    return model


model_outer = ModelTest()
model_outer.compile(loss="mse")
'''name自动重命名_n后与传入的name匹配，和_n没有关系，model中有重复使用重名的input会报错'''
model_outer.fit({"x1": np.array([1, 2, 3]).reshape((1, 3)), "x2": np.array([1, 2, 3]).reshape((1, 3)),
                 "x3": np.array([1, 2, 3])})
