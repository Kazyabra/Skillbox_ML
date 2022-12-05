import os
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    cwd = os.path.dirname(os.path.abspath(__file__))  # current work directory
    f_name = "statistic.csv"
    path = f"{cwd}\\data\\{f_name}"

    stat = pd.read_csv(path)
    print(stat.head())

    # Статистика 1-й линии (всего закрыто)
    fig = plt.figure()
    fig.suptitle('Статистика 1-й линии (всего закрыто)')
    stat.groupby('Дата').Засутки.sum().plot(kind='bar', x='Дата')
    plt.tight_layout()  # оптимизируем поля и расположение объектов
    plt.show()
