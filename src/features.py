def add_features(df):
    df = df.copy()

    # Общая площадь
    df["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]

    # Возраст дома на момент продажи
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    # Лет после ремонта на момент продажи
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    # Всего ванных
    df["TotalBath"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    # Общая площадь крыльца
    df["TotalPorch"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
    )

    # Есть ли 2-й этаж
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)

    # Есть ли гараж
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)

    # Есть ли подвал
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)

    return df
