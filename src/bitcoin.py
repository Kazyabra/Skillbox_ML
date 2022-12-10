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

    # добавил новый столбец на основе существующих
    bitcoin["hl_co_abs"] = ((bitcoin.high - bitcoin.low) / (bitcoin.close - bitcoin.open)).abs()
    # график по этому столбцу
    fig = plt.figure()
    fig.suptitle('BTC/USD график коэфф hl_co_abs')
    plt.plot(bitcoin.date, bitcoin.hl_co_abs, label='HL_CO_ABS')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Добавляем столбец - средняя цена открытия за предыдущие 7 дней
    #  Скользящее окно rolling(window=), shift смещение окна
    # Скользящее окно 7 предыдущих строчек таблицы(shift смещает окно на строку вниз,
    # чтобы не брать в расчет текущий день)
    bitcoin["open_mean_7d"] = bitcoin.open.shift(1).rolling(window=7).mean()

    bitcoin["max_close_14d"] = bitcoin.close.shift(1).rolling(window=14).max()
    bitcoin["min_close_14d"] = bitcoin.close.shift(1).rolling(window=14).min()
    print(bitcoin.head(20))

    # как сделать график между 2 датами
    okt2022 = bitcoin[bitcoin.date.between("2022-10-01", "2022-11-01")]
    fig = plt.figure()
    fig.suptitle('BTC/USD график open за октябрь 2022')
    plt.plot(okt2022.date, okt2022.open, label='open')
    plt.legend()
    plt.tick_params(axis='x', rotation=90)  # развернуть вертикально подписи по оси х
    plt.tight_layout()
    plt.show()

    # Задача: предсказать курс close на следующий день.
    # Добавить 7 колонок: цена close за 1 день назад (2,3,4,5,6,7 дней назад).
    for day in range(1, 8):
        name = f"close_{day}d"
        bitcoin[name] = bitcoin["close"].shift(day)

    # добавляем еще столбцы
    bitcoin["month"] = bitcoin["date"].dt.month
    bitcoin["year"] = bitcoin["date"].dt.year
    bitcoin["dayofweek"] = bitcoin["date"].dt.dayofweek

    # изменение датафреймов pandas
    # теперь нужно убрать все нечисловые столбцы, они не нужны модели
    # in_place=False по умолчанию текущий датафрейм не меняется
    bitcoin.drop("symbol", axis=1, inplace=True)  # удалить столбец из датафрейма
    bitcoin.drop("date", axis=1, inplace=True)  # удалить столбец из датафрейма

    # избавиться от NaN
    # dropna - удалить все строки с NaN
    # fillna - заполнить все пропуски (вперед/назад)
    # interpolate - интерполяция по соседним
    bitcoin.fillna(method="backfill", inplace=True)

    print(bitcoin.head(20))

    # x = вход, y = выход
    # x= данные за текущий день, y = close за след. день
    # target = close на завтра
    bitcoin["target"] = bitcoin.close.shift(-1)
    # в x положили все кроме target (без последней строки, т.к. там получится NaN)
    x = bitcoin[:-1].drop("target", axis=1)
    # в y положили target (без последней строки, т.к. там получится NaN)
    y = bitcoin[:-1].target

    # Train - тренировочная выборка = (x_train, y_train)
    # Test - тестовая выборка = (x_test, y_test)
    # Metric - средняя абсолютная ошибка, максимальная ошибка и др. - оценка качества

