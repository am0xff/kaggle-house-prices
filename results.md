# House Prices — Сравнение всех моделей

## Результаты

| Модель | CV RMSE | CV с FE | Kaggle |
|--------|---------|---------|--------|
| StackingRegressor (Lasso+Cat+XGB) | 0.1115 | 0.1115 | 0.12417 |
| VotingRegressor (Lasso+Cat+XGB) | 0.1126 | 0.1126 | 0.12341 |
| CatBoost (depth=6, lr=0.05) | 0.1169 | 0.1147 | 0.12340 |
| LinearRegression | 0.1164 | — | 0.14316 |
| Lasso (alpha=0.005) | 0.1397 | 0.1159 | — |
| ElasticNet (alpha=0.005, l1=0.9) | 0.1396 | — | — |
| Ridge (alpha=100) | 0.1401 | 0.1167 | — |
| XGBoost (n_est=500, depth=3, lr=0.05) | 0.1212 | 0.1218 | — |
| LightGBM (n_est=500, depth=4, lr=0.05) | 0.1225 | 0.1233 | — |
| Random Forest (500, depth=15) | 0.1361 | 0.1343 | — |
| DNN (BatchNorm, 1000 epochs) | 0.1441 | — | — |
| KNN (n=10, distance, manhattan) | 0.1684 | — | — |
| Decision Tree (max_depth=5) | 0.1918 | — | — |

## Лучший скор на Kaggle: 0.12340 (CatBoost)

## Выводы

**Линейные модели:** LinearRegression переобучается (CV=0.1164, Kaggle=0.14316). Lasso/Ridge/ElasticNet с регуляризацией стабильнее. Feature Engineering дал большой прирост линейным моделям (Lasso: 0.1397 → 0.1159).

**Деревья и KNN:** Decision Tree и KNN — слабейшие модели. Random Forest значительно лучше одиночного дерева.

**Бустинги:** CatBoost — лучший одиночный бустинг. XGBoost и LightGBM слабее на этом датасете.

**DNN:** Хуже бустингов на табличных данных (0.1441 vs 0.1147). Типичная ситуация — нейросети проигрывают градиентному бустингу на небольших табличных данных.

**Ансамбли:** Stacking лучший по CV (0.1115), но CatBoost лучший на Kaggle (0.12340). Смешивание разных типов моделей (линейная + бустинги) работает лучше чем однородные ансамбли.

**Feature Engineering:** Новые фичи (TotalSF, HouseAge, RemodAge, TotalBath, TotalPorch) помогли линейным моделям больше, чем бустингам. CatBoost сам умеет комбинировать фичи.

**CV vs Kaggle:** CV не всегда совпадает с лидербордом. Stacking — лучший по CV, но худший из трёх на Kaggle.