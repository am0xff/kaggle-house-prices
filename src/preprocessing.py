import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class HousePricesPreprocessor:
    """Stateful preprocessor: fit запоминает статистики из train, transform применяет."""

    # Пропуски тип 1: "нет объекта"
    NONE_COLS_CAT = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "MasVnrType",
    ]
    NONE_COLS_NUM = [
        "GarageYrBlt",
        "GarageArea",
        "GarageCars",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "MasVnrArea",
    ]

    # Пропуски тип 2: реальные (категориальные — модой)
    MODE_COLS = [
        "Electrical",
        "MSZoning",
        "Utilities",
        "Functional",
        "Exterior1st",
        "Exterior2nd",
        "KitchenQual",
        "SaleType",
    ]

    # Ordinal encoding: фиксированный порядок
    ORDINAL_COLS = {
        "ExterQual": ["Fa", "TA", "Gd", "Ex"],
        "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
        "BsmtQual": ["None", "Fa", "TA", "Gd", "Ex"],
        "BsmtCond": ["None", "Po", "Fa", "TA", "Gd"],
        "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
        "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
        "KitchenQual": ["Fa", "TA", "Gd", "Ex"],
        "Functional": ["Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
        "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
        "GarageFinish": ["None", "Unf", "RFn", "Fin"],
        "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
        "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
        "PavedDrive": ["N", "P", "Y"],
        "PoolQC": ["None", "Fa", "Gd", "Ex"],
        "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
        "LotShape": ["IR3", "IR2", "IR1", "Reg"],
        "LandSlope": ["Sev", "Mod", "Gtl"],
        "CentralAir": ["N", "Y"],
    }

    # One-hot encoding
    LOW_CARD_COLS = [
        "MSZoning",
        "Street",
        "Alley",
        "LandContour",
        "Utilities",
        "LotConfig",
        "BldgType",
        "MasVnrType",
        "Electrical",
        "MiscFeature",
    ]

    # Target encoding
    HIGH_CARD_COLS = [
        "Neighborhood",
        "Condition1",
        "Condition2",
        "HouseStyle",
        "RoofStyle",
        "RoofMatl",
        "Exterior1st",
        "Exterior2nd",
        "Foundation",
        "Heating",
        "GarageType",
        "SaleType",
        "SaleCondition",
    ]

    def __init__(self, scale=False):
        self.scale = scale
        # Запоминаем в fit:
        self.lotfrontage_median_ = None
        self.mode_values_ = {}
        self.target_means_ = {}
        self.onehot_columns_ = None
        self.scaler_ = None

    def fit(self, df, y=None):
        """Запоминает статистики из train."""
        # Медиана LotFrontage
        self.lotfrontage_median_ = df["LotFrontage"].median()

        # Моды для категориальных
        for col in self.MODE_COLS:
            self.mode_values_[col] = df[col].mode()[0]

        # Target encoding: средние по категориям (нужен y)
        if y is not None:
            for col in self.HIGH_CARD_COLS:
                self.target_means_[col] = y.groupby(df[col]).mean()

        return self

    def transform(self, df):
        """Применяет все преобразования."""
        df = df.copy()

        # 1. Пропуски: "нет объекта"
        for col in self.NONE_COLS_CAT:
            if col in df.columns:
                df[col] = df[col].fillna("None")
        for col in self.NONE_COLS_NUM:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 2. Пропуски: реальные
        df["LotFrontage"] = df["LotFrontage"].fillna(self.lotfrontage_median_)
        for col, value in self.mode_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)

        # 3. Ordinal encoding
        for col, order in self.ORDINAL_COLS.items():
            if col in df.columns:
                mapping = {val: i for i, val in enumerate(order)}
                df[col] = df[col].map(mapping)

        # 4. One-hot encoding
        df = pd.get_dummies(df, columns=self.LOW_CARD_COLS, drop_first=True)
        if self.onehot_columns_ is None:
            self.onehot_columns_ = df.columns.tolist()
        else:
            # Выравниваем колонки с train
            for col in self.onehot_columns_:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.onehot_columns_]

        # 5. Target encoding
        for col, means in self.target_means_.items():
            if col in df.columns:
                global_mean = means.mean()
                df[col] = df[col].map(means).fillna(global_mean)

        # 6. Нормализация (опционально)
        if self.scale:
            num_cols = df.select_dtypes(include="number").columns
            if self.scaler_ is None:
                self.scaler_ = StandardScaler()
                df[num_cols] = self.scaler_.fit_transform(df[num_cols])
            else:
                df[num_cols] = self.scaler_.transform(df[num_cols])

        # Конвертируем bool в int
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

        return df

    def fit_transform(self, df, y=None):
        """fit + transform в одном вызове."""
        self.fit(df, y)
        return self.transform(df)
