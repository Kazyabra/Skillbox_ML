import os
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    cwd = os.path.dirname(os.path.abspath(__file__))  # current work directory
    f_name = "Bitfinex_BTCUSD_d.csv"
    path = f"{cwd}\\data\\{f_name}"

    # убрал лишний столбец, столбец с датой и временем установил как дату, отсортировал по дате
    bitcoin = pd.read_csv(path, parse_dates=['date']).drop('unix', axis=1).sort_values(by='date')
    # сбросил индекс(переиндексировал) а то сперва шли свежие даты в индексе, потом старые
    bitcoin.reset_index(drop=True, inplace=True)
    pd.set_option("display.max.columns", None)  # не скрывать столбцы
    print(bitcoin.head())
    print(bitcoin.date.describe())

    # BTC/USD open price
    fig = plt.figure()
    fig.suptitle('BTC/USD open price')
    bitcoin.open.plot()
    plt.tight_layout()  # оптимизируем поля и расположение объектов
    plt.show()

    # BTC/USD графики по всем колонкам
    fig = plt.figure()
    fig.suptitle('BTC/USD графики по всем колонкам')
    plt.plot(bitcoin.date, bitcoin.open, label='Open')
    plt.plot(bitcoin.date, bitcoin.close, label='Close')
    plt.plot(bitcoin.date, bitcoin.high, label='High')
    plt.plot(bitcoin.date, bitcoin.low, label='Low')
    plt.legend()  # печать легенды
    plt.tight_layout()  # оптимизируем поля и расположение объектов
    plt.show()
