# Финальный проект по практикуму осень 2021.
## Ансамбли алгоритмов. Веб-сервер.
## Введение


Проект делится на две логические части:
* Экспериментальная
* Инфраструктурная

Экспериментальная часть подразумевает реализацию 
таких алгоритмов ансамблирования, как
[`RandomForest`](http://www.machinelearning.ru/wiki/images/3/3a/Voron-ML-Compositions1-slides.pdf) 
и [`GradientBoosting`](http://www.machinelearning.ru/wiki/images/2/21/Voron-ML-Compositions-slides2.pdf), 
их качественное сравнение, анализ 
экспериментов и написание отчета в системе [`LaTeX`](https://www.latex-project.org/).

Вторая часть - инфраструктурная. Необходимо поднять 
свой веб-сервер с помощью 
[`Docker`](https://www.docker.com/) и [`Flask`](https://flask.palletsprojects.com/en/latest/), а также реализовать
возможность загрузки данных в формате  `.csv` 
и обучения на них моделей машинного обучения.

---


## Структура проекта:

```angular2html
*
|
| --> src/
|       | --> ensembles.py
|       |       | --> RandomForestRMSE
|       |       | --> GradientBoostingRMSE
|       | --> start_flask.py
|       | --> flask_utils.py
|       | --> static/
|       |       | --> css/
|       |       |       | --> (...CSS files...)
|       |       | --> datasets/
|       |       | --> img/
|       |       |       | --> bg.svg
|       |       | --> model/
|       | --> templates/
|       |       | --> (...HTML files...)
|       
| --> scripts/
|       | --> build.sh
|
| --> Dockerfile
| --> experiments.ipynb
| --> Report.pdf
| --> README.md
| --> requirements.txt
| --> run.sh
|
*
```

---

**Важнo!** Для запуска проекта необходимо иметь систему `Docker` на локальном компьютере. Если у Вас её ещё нет, 
подробнее можно узнать [тут](https://docs.docker.com/get-docker/).

---

Запуск системы возможен как на архитектуре ARM64, так и на AMD64.

---
## Установка и запуск

Для начала работы необходимо клонировать репозиторий:
```angular2html
$ git clone https://github.com/KuzminAleksandr/TheLastProject.git
```

Далее необходимо взять готовый образ с `Docker Hub`:
```angular2html
$ docker pull kuzminalexcmc/dockerhub:docker_flask   
```
Теперь осталось лишь запустить образ. Для этого перейдите в директорию клонированного репозитория и выполните:
```angular2html
$ bash run.sh
```
Готово! Сервер должен запуститься. [Нажмите](http://0.0.0.0:5001/), чтобы перейти к веб-приложению.

Для того чтобы закончить работу приложения нажмите `CTRL + C`.

---

## Функционал

Веб-приложение дает возможность:
* Выбрать алгоритм ML: `RandomForest` или `GradientBoosting`.
* Загрузить тренировочный и валидационный датасет в формате `.csv`.
* Подбирать параметры: `n_estimators`, `max_depth`, `feature_subsample_size`, а также `learning_rate`.
* Получить обученную модель и логи обучения.
* Сделать предсказания на тестовом множестве.

---

Подробнее о `RandomForest` и `GradientBoosting` можно узнать 
[тут](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble).

---

## Структура данных

Для корректной работы системы необходимо:
* Все данные должны иметь формат `.csv`
* Наличие столбца `target` в тренировочном и валидационном датасете.
* Совпадение множества признаков в тренировочном и валидационном датасете 
и сохранение порядка их следования.
* Все признаки должны быть числовыми: 
предобработка категориальных признаков, строк и т.д. 
предоставляются пользователю.
* Тестовое множество должно иметь идентичные 
тренировочному множеству признаки, исключая целевую переменную.

Общий вид датасета:\
`| feature_column_1 | feature_column_2 | ... | feature_column_n | target (optional) |`


