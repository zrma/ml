from random import randint
import numpy as np
import pandas as pd


def run():
    arr = pd.Series((randint(0, 10) for _ in range(10)))

    print(arr)
    print(arr.values)
    print(arr.index)
    print(arr.dtypes)

    df = pd.DataFrame(arr)
    print(df)

    df = df.apply(lambda x: x * 10, axis=1)
    print(df)
    print(df.index)
    print(df.values)
    print(df.columns)

    data = {'name': ['Beomwoo', 'Beomwoo', 'Beomwoo', 'Kim', 'Park'],
            'year': [2013, 2014, 2015, 2016, 2015],
            'points': [1.5, 1.7, 3.6, 2.4, 2.9]}
    df = pd.DataFrame(data)
    print(df)


if __name__ == "__main__":
    run()
