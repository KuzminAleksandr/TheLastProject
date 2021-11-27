# Финальный проект по практикуму осень 2021.
## Ансамбли алгоритмов. Веб-сервер.
### Введение


Проект делится на две логические части:
* Экспериментальная
* Инфраструктурная

Экспериментальная часть подразумевает реализацию 
таких алгоритмов ансамблирования, как
[`RandomForest`](http://www.machinelearning.ru/wiki/images/3/3a/Voron-ML-Compositions1-slides.pdf) 
и [`GradientBoosting`](http://www.machinelearning.ru/wiki/images/2/21/Voron-ML-Compositions-slides2.pdf), 
их качественное сравнение, анализ 
экспериментов и написание отчета в системе `LaTeX`.

Вторая часть - инфраструктурная. Необходимо поднять 
свой веб-сервер с помощью `Docker` и `Flask`, а также реализовать
возможность загрузки данных в формате  `.csv` 
и обучения на них моделей машинного обучения.

---


### Структура проекта:

```angular2html
*
|
| --> src/
|       | --> ensembles.py
|       |       | --> RandomForestRMSE
|       |       | --> GradientBoostingRMSE
|
| --> scripts/
|       | --> build.sh
|       | --> run.sh
|
| --> Dockerfile
| --> Report.pdf
| --> README.md
|
*
```

---

