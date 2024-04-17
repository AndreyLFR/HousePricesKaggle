import pandas as pd
import pprint
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

df_train = pd.read_csv('train.csv')

#определяю отсутствующие значения в df
#print(df_train.info())
dict_nan = {}
count_row = df_train.shape[0]
print(count_row)
for col in list(df_train):
    count_nan = df_train.isnull().sum()[col]
    if count_nan:
        dict_nan[col] = [count_nan, round(count_nan/count_row*100, 2)]
print('--------')
print('Значения Nan: ')
pprint.pprint(dict_nan)

#заполняю отсутствующие данные в df
#удалим атрибуты, у которых нет более 40% значений, так как выводы сделанные на их базе могут дать ошибку
print('-----')
print(f'Было: {df_train.shape}')
list_col_del = []
for col in dict_nan:
    if dict_nan[col][1] > 40:
        del df_train[col]
        list_col_del.append(col)
print(f'удален столбец {list_col_del}')
print(f'Стало: {df_train.shape}')

#заполняю отсутствующие значения
for key in list_col_del:
    del dict_nan[key]

dict_nan_with_type = {}
for key, value in dict_nan.items():
    dict_nan_with_type[key] = (value, df_train.dtypes[key], df_train[key].nunique())
print('--------')
print('Значения Nan: ')
pprint.pprint(dict_nan_with_type)

#количественные атрибуты заполню медианным значением
quantitative = ['GarageYrBlt', 'LotFrontage', 'MasVnrArea']
df_train[quantitative] = df_train[quantitative].fillna(df_train[quantitative].median())
print('------')
print('Проверка заполнения медианой Nan: ')
print(df_train.isnull().sum()[quantitative])

#заполнение качественных атрибутов модой
for key in quantitative:
    del dict_nan_with_type[key]

dict_nan_with_type_unique = {}
for key, value in dict_nan_with_type.items():
    dict_nan_with_type_unique[key] = (value, df_train[key].unique())

print('--------')
print('Перепроверю вдруг количественные переменные имеют тип Object')
print('Значения Nan: ')
pprint.pprint(dict_nan_with_type_unique)

quality = list(dict_nan_with_type_unique)
df_train[quality] = df_train[quality].fillna(df_train[quality].mode())

for col in quality:
    mode_ = df_train[col].mode()[0]  # Вычисление моды (наиболее часто встречающегося значения)
    df_train[col] = df_train[col].fillna(mode_)

print('------')
print('Проверка заполнения модой Nan: ')
print(df_train.isnull().sum()[quality])
print('------')
print('Результат: ')
print(df_train.info())

#Проверяю и убираю дубли
print('------')
print('удаление дубликатов')
print(df_train.shape)
df_train = df_train.drop_duplicates()
print(df_train.shape)

#EDA
#распределение target
plt.hist(df_train['SalePrice'], color='blue', edgecolor='black', bins=100000)
plt.title('Частотность цены')
plt.xlabel('Price (100тыс)')
plt.ylabel('Count')
plt.show()

print(df_train.describe())
limit_sale_price = df_train['SalePrice'].mean() + 3 * df_train['SalePrice'].std()
print(df_train.shape)
df_train = df_train[df_train['SalePrice'] < limit_sale_price]
print(df_train.shape)

#проверка корреляции между переменными
for col in quantitative:
    if col != 'SalePrice':
        print(pearsonr(df_train['SalePrice'], df_train[col]))

#OverallQual
plt.scatter(df_train['OverallQual'], df_train['SalePrice'])
plt.xlabel('Оценка качества')
plt.ylabel('Цена')
plt.title('Диаграмма зависимости цены от OverallQual')
plt.show()

#GrLivArea
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('Цена')
plt.title('Диаграмма зависимости цены от GrLivArea')
plt.show()

#GarageCars
plt.scatter(df_train['GarageCars'], df_train['SalePrice'])
plt.xlabel('GarageCars')
plt.ylabel('Цена')
plt.title('Диаграмма зависимости цены от GarageCars')
plt.show()

#преобразую качественные переменные в количественные
df = pd.get_dummies(df_train, columns=quality, drop_first=True)
print(df.head())

df.to_csv('cleaned_house_prices', encoding='utf-8', index=False)
