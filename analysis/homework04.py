import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

"""
Dataset obsahuje nasledujici promenne:
 'Age' - vek v rocich
 'Fare' - cena jizdenky
 'Name' - jmeno cestujiciho
 'Parch' - # rodicu/deti daneho cloveka na palube
 'PassengerId' - Id
 'Pclass' - Trida, 1 = 1. trida, 2 = 2.trida, 3 = 3.trida
 'Sex' - pohlavi
 'SibSp' - # sourozencu/manzelu daneho cloveka na ppalube
 'Survived' - 0 = Neprezil, 1 = Prezil
 'Embarked' - Pristav, kde se dany clovek nalodil. C = Cherbourg, Q = Queenstown, S = Southampton
 'Cabin' - Cislo kabiny
 'Ticket' - Cislo tiketu
"""



def load_dataset(train_file_path, test_file_path):
    """
    Napiste funkci, ktera nacte soubory se souboru zadanych parametrem a vytvori dva separatni DataFrame. Pro testovani vyuzijte data 'data/train.csv' a 'data/test.csv'
    Ke kazdemu dataframe pridejte sloupecek pojmenovaný jako "Label", ktery bude obsahovat hodnoty "Train" pro train.csv a "Test" pro test.csv.

    1. Pote slucte oba dataframy.
    2. Z vysledneho slouceneho DataFramu odstraňte sloupce  "Ticket", "Embarked", "Cabin".
    3. Sloučený DataDrame bude mít index od 0 do do počtu řádků.
    4. Vratte slouceny DataDrame.
    """
    data_train = pd.read_csv(train_file_path)
    data_test = pd.read_csv(test_file_path)
    data_train.insert(0, "Label", "Train")
    data_test.insert(0, "Label", "Test")
    result = data_train.append(data_test, ignore_index=True)
    result = result.drop(["Ticket", "Embarked", "Cabin"], axis='columns')
    return result


def get_missing_values(df : pd.DataFrame) -> pd.DataFrame:
    """
    Ze zadaneho dataframu zjistete chybejici hodnoty. Vytvorte DataFrame, ktery bude obsahovat v indexu jednotlive promenne
    a ve prvnim sloupci bude promenna 'Total' obsahujici celkovy pocet chybejicich hodnot a ve druhem sloupci promenna 'Percent',
    ve ktere bude procentualni vyjadreni chybejicich hodnot vuci celkovemu poctu radku v tabulce.
    DataFrame seradte od nejvetsich po nejmensi hodnoty.
    Vrattre DataFrame chybejicich hodnot a celkovy pocet chybejicich hodnot.

    Priklad:

               |  Total  |  Percent
    "Column1"  |   34    |    76
    "Column2"  |   0     |    0

    """

    result = pd.DataFrame(df.isna().sum(), columns=["Total"])
    result["Percent"] = result["Total"] / len(df.index) * 100
    result = result.sort_values(by="Total", ascending=False)
    return result

def substitute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chybejici hodnoty ve sloupecku "Age" nahradte meanem hodnot z "Age".
    Chybejici hodnoty ve sloupecku "Fare" nahradte meadianem hodnot z "Fare".
    V jednom pripade pouzijte "loc" a ve druhem "fillna".
    Zadany DataFrame neupravujte, ale vytvorte si kopii.
    Vratte upraveny DataFrame.
    """
    result = df
    age_mean = np.nanmean(df["Age"].values)
    result["Age"].loc[pd.isna(result["Age"])] = age_mean
    fare_median = np.nanmedian(result["Fare"].values)
    result["Fare"] = result["Fare"].fillna(fare_median)
    return result



def get_correlation(df: pd.DataFrame) -> float:
    """
    Spocitejte korelaci pro "Age" a "Fare" a vratte korelaci mezi "Age" a "Fare".
    """
    return df["Age"].corr(df["Fare"])

def get_survived_per_class(df : pd.DataFrame, group_by_column_name : str) ->pd.DataFrame:
    """
    Spocitejte prumer z promenne "Survived" pro kazdou skupinu zadanou parametrem "group_by_column_name".
    Hodnoty seradte od nejvetsich po mejmensi.
    Hodnoty "Survived" zaokhroulete na 2 desetinna mista.
    Vratte pd.DataFrame.

    Priklad:

    get_survived_per_class(df, "Sex")

                 Survived
    Male     |      0.32
    Female   |      0.82

    """
    result = pd.DataFrame(list(df[group_by_column_name].values), columns=[group_by_column_name])
    result["Survived"] = list(df["Survived"].values)
    result = round(result.groupby(group_by_column_name).mean(), 2)
    result = result.reset_index(level=[group_by_column_name])
    return result





def get_outliers(df: pd.DataFrame) -> (int, str):
    """
    Vyfiltrujte odlehle hodnoty (outliers) ve sloupecku "Fare" pomoci metody IRQ.
    Tedy spocitejte rozdil 3. a 1. kvantilu, tj. IQR = Q3 - Q1.
    Pote odfiltrujte vsechny hodnoty nesplnujici: Q1 - 1.5*IQR < "Fare" < Q3 + 1.5*IQR.
    Namalujte box plot pro sloupec "Fare" pred a po vyfiltrovani outlieru.
    Vratte tuple obsahujici pocet outlieru a jmeno cestujiciho pro nejvetsi outlier.
    """
    Q1 = df.quantile(0.25)["Fare"]
    Q3 = df.quantile(0.75)["Fare"]
    IQR = Q3 - Q1
    df_out = df[~((df["Fare"] < (Q1 - 1.5 * IQR)) | (df["Fare"] > (Q3 + 1.5 * IQR)))]
    df.boxplot(column="Fare")
    plt.show()
    df_out.boxplot(column="Fare")
    plt.show()
    return (len(df) - len(df_out), df.iloc[df.index[df["Fare"] == max(list(df["Fare"]))].tolist()[0]]["Name"])

def f(d, col):
    d[col] = (d[col] - d[col].min()) / (d[col].max() - d[col].min())
    return d

def normalise(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Naskalujte sloupec "col" zadany parametrem pro kazdou "Pclass" hodnotu z dataframu "df" zvlast.
    Pouzijte vzorec: scaled_x_i = (x_i - min(x)) / (max(x) - min(x)), kde "x_i" prestavuje konkretni hodnotu ve sloupeci "col".
    Vratte naskalovany dataframe.
    """
    df = df.groupby("Pclass").apply(f, col)
    return df





def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vytvorte 3 nove promenne:
    1. "Fare_scaled" - vytvorte z "Fare" tak, aby mela nulovy prumer a jednotkovou standartni odchylku.
    2. "Age_log" - vytvorte z "Age" tak, aby nova promenna byla logaritmem puvodni "Age".
    3. "Sex" - Sloupec "Sex" nahradte: "female" -> 1, "male" -> 0, kde 0 a 1 jsou integery.

    Nemodifikujte predany DataFrame, ale vytvorte si novy, upravte ho a vratte jej.
    """
    new_df = pd.DataFrame(df)
    arr = np.array(df["Fare"].values)
    std_fare = np.nanstd(arr)
    mean_fare = np.nanmean(arr)
    new_df["Fare_scaled"] = (df["Fare"] - mean_fare) / std_fare
    new_df["Age_log"] = np.log(df["Age"].values)
    new_df["Sex"] = np.where(new_df["Sex"] == "female", 1, 0)
    return new_df


def my_func(df):
    df["Survival_rate"] = df["Survived"]["count"] / df["Survived"]["size"]
    return df

def determine_survival(df: pd.DataFrame, n_interval: int, age: float, sex: str) -> float:
    """
    Na zaklade statistickeho zpracovani dat zjistete pravdepodobnost preziti Vami zadaneho cloveka (zadava se vek a pohlavi pomoci parametru "age" a "sex")

    Vsechny chybejici hodnoty ve vstupnim DataFramu ve sloupci "Age" nahradte prumerem.
    Rozdelte "Age" do n intervalu zadanych parametrem "n_interval". Napr. pokud bude Age mit hodnoty [2, 13, 18, 25] a mame jej rozdelit do 2 intervalu,
    tak bude vysledek:

    0    (1.977, 13.5]
    1    (1.977, 13.5]
    2     (13.5, 25.0]
    3     (13.5, 25.0]

    Pridejte k rozdeleni jeste pohlavi. Tj. pro kazdou kombinaci pohlavi a intervalu veku zjistete prumernou
    pravdepodobnost preziti ze sloupce "Survival" a tu i vratte.

    Vysledny DataFrame:

    "AgeInterval"   |    "Sex"    |   "Survival Probability"
       (0-10)       | "male"      |            0.21
       (0-10)       | "female"    |            0.28
       (10-20)      | "male"      |            0.10
       (10-20)      | "female"    |            0.15
       atd...

    Takze vystup funkce determine_survival(df, n_interval=20, age = 5, sex = "male") bude 0.21. Tato hodnota bude navratovou hodnotou funkce.

    """
    df["Age"] = df["Age"].fillna(np.nanmean(df["Age"].values))
    df = df[["Age", "Sex", "Survived"]].groupby([pd.cut(df["Age"].sort_values(), n_interval), "Sex"]).agg(["sum", "count"])
    df = df.drop("Age", axis=1)
    df = df.reset_index()
    df["Survival_rate"] = df["Survived"]["sum"] / df["Survived"]["count"]
    df["Left"] = pd.IntervalIndex(df["Age"]).left
    df["Right"] = pd.IntervalIndex(df["Age"]).right
    prob = df[(df["Left"] < age) & (age <= df["Right"]) & (df["Sex"] == sex)]["Survival_rate"].values
    return prob if prob else 0



data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
df = load_dataset(os.path.join(data_dir, 'train.csv'), os.path.join(data_dir, 'test.csv'))

