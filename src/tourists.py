# Интенсив от Skillbox по машинному обучению https://www.youtube.com/watch?v=TvhDbkhuo5c

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    cwd = os.path.dirname(os.path.abspath(__file__))  # current work directory
    f_name = "trips_data.xlsx"
    path = f"{cwd}\\data\\{f_name}"

    trips_data = pd.read_excel(path)
    pd.set_option("display.max.columns", None)  # не скрывать столбцы
    print(trips_data)

    # распределение по зарплате
    fig = plt.figure()
    fig.suptitle('Распределение по зарплате')
    trips_data.salary.hist()
    plt.tight_layout()  # оптимизируем поля и расположение объектов
    plt.show()

    # распределение по возрасту
    fig = plt.figure()
    fig.suptitle('Распределение по возрасту')
    trips_data.age.hist()
    plt.tight_layout()
    plt.show()

    # распределение по отдыху
    print('Распределение по отдыху')
    print(trips_data.vacation_preference.value_counts())
    fig = plt.figure()
    fig.suptitle('Распределение по отдыху')
    trips_data.vacation_preference.value_counts().plot(kind='bar')
    plt.tight_layout()
    plt.show()

    # распределение по городам
    print('Распределение по городам')
    print(trips_data.city.value_counts())
    fig = plt.figure()
    fig.suptitle('Распределение по городам')
    trips_data.city.value_counts().plot(kind='bar')
    plt.tight_layout()
    plt.show()

    # распределение по транспорту
    print('Распределение по транспорту')
    print(trips_data.transport_preference.value_counts())
    fig = plt.figure()
    fig.suptitle('Распределение по транспорту')
    trips_data.transport_preference.value_counts().plot(kind='bar')
    plt.tight_layout()
    plt.show()

    # распределение по количеству членов семьи
    print('Распределение по количеству членов семьи')
    print(trips_data.family_members.value_counts())
    fig = plt.figure()
    fig.suptitle('Распределение по количеству членов семьи')
    trips_data.family_members.value_counts().plot(kind='bar')
    plt.tight_layout()
    plt.show()

    # распределение по месту путешествия
    print('Распределение по месту путешествия')
    print(trips_data.target.value_counts())
    fig = plt.figure()
    fig.suptitle('Распределение по месту путешествия')
    trips_data.target.value_counts().plot(kind='bar')
    plt.tight_layout()
    plt.show()

    # Категоризация: нужно предсказать куда хочет поехать человек
    # Входные данные X - всё кроме target
    # Выходные данные Y - target
    x = trips_data.drop("target", axis=1)
    y = trips_data.target
    # переводим категориальные столбцы в бинарные
    x_dummies = pd.get_dummies(x, columns=["city", "vacation_preference", "transport_preference"])
    print(f'Форма входных данных: {x_dummies.shape}')

    # Создаем модель машинного обучения
    model = RandomForestClassifier()
    # обучить модель
    model.fit(x_dummies, y)

    # Поверхностная оценка качества (accuracy / train dataset)
    # Насколько хорошо модель делает предсказания
    print(f'Качество предсказаний модели: {model.score(x_dummies, y)}')

    # печатаю шаблон новой записи для датафрейма
    # print({col: [0] for col in x_dummies.columns})
    # полученный шаблон заполняю значениями (ручками)
    example = {'salary': [30000],
               'age': [43],
               'family_members': [2],
               'city_Екатеринбург': [0],
               'city_Киев': [0],
               'city_Краснодар': [0],
               'city_Минск': [0],
               'city_Москва': [1],
               'city_Новосибирск': [0],
               'city_Омск': [0],
               'city_Петербург': [0],
               'city_Томск': [0],
               'city_Хабаровск': [0],
               'city_Ярославль': [0],
               'vacation_preference_Архитектура': [0],
               'vacation_preference_Ночные клубы': [0],
               'vacation_preference_Пляжный отдых': [1],
               'vacation_preference_Шоппинг': [0],
               'transport_preference_Автомобиль': [0],
               'transport_preference_Космический корабль': [0],
               'transport_preference_Морской транспорт': [0],
               'transport_preference_Поезд': [0],
               'transport_preference_Самолет': [1]}

    # закидываю словарь в датафрейм с названиями столбцов от x_dummies
    example_df = pd.DataFrame(example, columns=x_dummies.columns)

    # Как сделать прогноз:
    # model.predict
    # model.predict_proba

    # получаю прогноз для нового человека(датафрейма example_df)
    print(model.classes_)  # печать классов модели
    print(model.predict_proba(example_df))  # распределение вероятности назначения классов
    print(f'Куда рекомендовать поехать человеку: {model.predict(example_df)}\n')

    # У кого из любителей самолетов больше членов семьи
    print('Фильтр по любителям Самолетов')
    print(trips_data.transport_preference == "Самолет")
    fig = plt.figure()
    fig.suptitle('Распределение любителей самолета по количеству членов семьи')
    trips_data[trips_data.transport_preference == "Самолет"].family_members.hist()
    # или можно так
    # trips_data[trips_data.transport_preference == "Самолет"].family_members.value_counts().plot(kind='bar')
    plt.tight_layout()
    plt.show()

    # Найти самых взрослых людей в каждом городе
    print("Найти самых взрослых людей в каждом городе")
    print(trips_data.groupby("city")["age"].max())

    # Средняя зарплата по городам
    print("Средняя зарплата по городам")
    print(trips_data.groupby("city")["salary"].mean())

    # Какой диапазон возрастов (20-30, 30-40, 40-50, 50-60, 60-70, 70-80) имеет самую высокую среднюю зарплату
    # делаем колонку с диапазонами возрастов
    trips_data["age_band"] = trips_data["age"] // 10
    print(trips_data.groupby("age_band")["salary"].mean().astype(int))
    fig = plt.figure()
    fig.suptitle('Распределение зарплаты под диапазонам возраста')
    plt.bar(range(1, 9), trips_data.groupby("age_band")["salary"].mean().astype(int))
    plt.tight_layout()
    plt.show()

    # Кто предпочитает Архитектуру, люди с высокой зарплатой или с низкой
    fig = plt.figure()
    fig.suptitle('Распределение любителей архитектуры по зарплате')
    trips_data[trips_data.vacation_preference == "Архитектура"].salary.hist()
    plt.tight_layout()
    plt.show()
