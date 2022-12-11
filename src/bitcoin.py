import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, max_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer

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
    # bitcoin["open_max_7d"] = bitcoin.open.shift(1).rolling(window=7).max()
    # bitcoin["open_min_7d"] = bitcoin.open.shift(1).rolling(window=7).min()

    bitcoin["close_mean_14d"] = bitcoin.close.shift(1).rolling(window=14).mean()
    # bitcoin["close_max_14d"] = bitcoin.close.shift(1).rolling(window=14).max()
    # bitcoin["close_min_14d"] = bitcoin.close.shift(1).rolling(window=14).min()

    bitcoin["close_mean_30d"] = bitcoin.close.shift(1).rolling(window=30).mean()
    # bitcoin["close_max_30d"] = bitcoin.close.shift(1).rolling(window=30).max()
    # bitcoin["close_min_30d"] = bitcoin.close.shift(1).rolling(window=30).min()

    bitcoin["close_mean_90d"] = bitcoin.close.shift(1).rolling(window=90).mean()
    # bitcoin["close_max_90d"] = bitcoin.close.shift(1).rolling(window=90).max()
    # bitcoin["close_min_90d"] = bitcoin.close.shift(1).rolling(window=90).min()

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
    bitcoin["day"] = bitcoin["date"].dt.day
    bitcoin["dayofweek"] = bitcoin["date"].dt.dayofweek

    # изменение датафреймов pandas
    # теперь нужно убрать все нечисловые столбцы, они не нужны модели
    # in_place=False по умолчанию текущий датафрейм не меняется
    bitcoin.drop("symbol", axis=1, inplace=True)  # удалить столбец из датафрейма
    bitcoin.drop("date", axis=1, inplace=True)  # удалить столбец из датафрейма

    # в столбце hl_co_abs есть значения с бесконечностью, надо от них избавиться
    # заменяем их на NaN
    bitcoin.replace([np.inf, -np.inf], np.nan, inplace=True)
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

    print(x.shape, y.shape)

    # разделяем данные на тренировочные и тестовые
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # Стандартизация - среднее значение в столбце = 0
    # scaler = StandardScaler().fit(x_train)
    # scaler.transform(x_train)
    # scaler.transform(x_test)
    # Нормализация - приведение значений к единому масштабу
    # scaler = Normalizer().fit(x_train)
    # scaler.transform(x_train)
    # scaler.transform(x_test)

    # обучаем модель Линейной регрессии на тренировочных данных
    print("Модель LinearRegression")
    model = LinearRegression()
    # model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
    model.fit(x_train, y_train)

    # проверяем коэф. детерминации на обучающих и тестовых данных
    print(f"Коэф. детерминации:\nОбучение {model.score(x_train, y_train)},\nТест {model.score(x_test, y_test)}")
    # получаем предсказания
    y_pred = model.predict(x_test)
    # Метрика средняя абсолютная ошибка Mean Absolute Error
    # print((y_pred - y_test).abs().mean())
    print(f"MAE = {mean_absolute_error(y_pred, y_test)}")  # то же самое
    # Метрика максимальная ошибка
    # print((y_pred - y_test).abs().max())
    print(f"MaxErr = {max_error(y_pred, y_test)}")  # то же самое

    # Обучаем модель рандомфорест регрессор
    print("Модель RandomForestRegressor")
    model = RandomForestRegressor(n_estimators=15, max_depth=15, random_state=42)
    model.fit(x_train, y_train)

    # проверяем коэф. детерминации на обучающих и тестовых данных
    print(f"Коэф. детерминации:\nОбучение {model.score(x_train, y_train)},\nТест {model.score(x_test, y_test)}")
    # получаем предсказания
    y_pred = model.predict(x_test)
    # Метрика средняя абсолютная ошибка Mean Absolute Error
    print(f"MAE = {mean_absolute_error(y_pred, y_test)}")
    # Метрика максимальная ошибка
    print(f"MaxErr = {max_error(y_pred, y_test)}")

    new_df = pd.concat([bitcoin.open.reset_index(drop=True),
                        bitcoin.year.reset_index(drop=True),
                        bitcoin.month.reset_index(drop=True),
                        bitcoin.day.reset_index(drop=True)], axis=1)
    new_df["date"] = new_df.year + new_df.month / 12
    # print(new_df.head(30))
    fig = plt.figure()
    sns.set_style('whitegrid')
    sns.lmplot(new_df, x="date", y="open", hue="year", palette='plasma', scatter_kws={'s': 0.3})

    plt.tight_layout()  # оптимизируем поля и расположение объектов
    plt.show()
