# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


PRODUCTS_FOLDER = 'Products/'
TRANSACTIONS_FOLDER = 'Transactions/'

categories_df = pd.read_csv(os.path.join(PRODUCTS_FOLDER, 'Categories.csv'), sep='|', header=None, names=['category_id', 'category_name'])

product_category_df = pd.read_csv(os.path.join(PRODUCTS_FOLDER, 'ProductCategory.csv'), sep='|')

transaction_files = glob(os.path.join(TRANSACTIONS_FOLDER, '*_Tran.csv'))
transactions_dfs = []

for file in transaction_files:
    df = pd.read_csv(file, sep='|', header=None, names=['date', 'store_id', 'customer_id', 'products'])
    df['products'] = df['products'].str.split(' ')  # Convertir la cadena de productos en lista
    transactions_dfs.append(df)
    

all_transactions_df = pd.concat(transactions_dfs, ignore_index=True)

# Convertir listas de productos a tuplas para que sean hashables (para chequeo de duplicados)
all_transactions_df['products'] = all_transactions_df['products'].apply(tuple)

all_transactions_df['date'] = pd.to_datetime(all_transactions_df['date'])

all_transactions_df['num_products'] = all_transactions_df['products'].apply(len)

# expandir los productos en filas nuevas
exploded_products = all_transactions_df.explode('products')
exploded_products['product_id'] = pd.to_numeric(exploded_products['products'], errors='coerce')  # Convertir a numérico

# Merge con categorías de productos (asumiendo que 'v.Code_pr' es el product_id)
exploded_products = exploded_products.merge(product_category_df, left_on='product_id', right_on='v.Code_pr', how='left')
exploded_products.rename(columns={'v.code': 'category_id'}, inplace=True)

# Merge con nombres de categorías
exploded_products = exploded_products.merge(categories_df, on='category_id', how='left')


print("\n### Revisión Inicial ###\n")

# Categories.csv
print("Categories.csv:")
print(f"Número de registros: {categories_df.shape[0]}")
print("Tipos de datos:\n", categories_df.dtypes)
print("Valores faltantes:\n", categories_df.isnull().sum())
duplicates_categories = categories_df.duplicated().sum()
print(f"Duplicados: {duplicates_categories}")

# ProductCategory.csv
print("\nProductCategory.csv:")
print(f"Número de registros: {product_category_df.shape[0]}")
print("Tipos de datos:\n", product_category_df.dtypes)
print("Valores faltantes:\n", product_category_df.isnull().sum())
duplicates_product_cat = product_category_df.duplicated().sum()
print(f"Duplicados: {duplicates_product_cat}")

# Transacciones combinadas
print("\nTransacciones combinadas:")
print(f"Número de registros: {all_transactions_df.shape[0]}")
print("Tipos de datos:\n", all_transactions_df.dtypes)
print("Valores faltantes:\n", all_transactions_df.isnull().sum())
duplicates_transactions = all_transactions_df.duplicated().sum()
print(f"Duplicados: {duplicates_transactions}")


# %%
print("\n### Estadísticas Descriptivas ###\n")

print("Variables Numéricas (Transacciones):")
num_stats = all_transactions_df[['num_products']].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
print(num_stats)

# Media, Mediana, Desviación Estándar
mean_num = all_transactions_df['num_products'].mean()
median_num = all_transactions_df['num_products'].median()
std_num = all_transactions_df['num_products'].std()
print(f"Media: {mean_num}, Mediana: {median_num}, Desviación Estándar: {std_num}")

# Detección de Outliers (usando método IQR)
Q1 = all_transactions_df['num_products'].quantile(0.25)
Q3 = all_transactions_df['num_products'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = all_transactions_df[
    (all_transactions_df['num_products'] < lower_bound) |
    (all_transactions_df['num_products'] > upper_bound)
]

print(f"Número de outliers en num_products (IQR): {len(outliers)}")
print(f"Rango permitido: [{lower_bound:.2f}, {upper_bound:.2f}]")

# Visualización de outliers (boxplot)
plt.figure(figsize=(8, 6))
sns.boxplot(x=all_transactions_df['num_products'])
plt.title('Boxplot de Número de Productos por Transacción (Detección de Outliers con IQR)')
plt.show()

# Otras numéricas derivadas: Frecuencia por cliente (número de transacciones por customer_id)
customer_freq = all_transactions_df['customer_id'].value_counts().reset_index(name='freq')
customer_freq.rename(columns={'customer_id': 'customer_id'}, inplace=True)  
print("\nEstadísticas de Frecuencia por Cliente:")
print(customer_freq['freq'].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]))

# Variables Categóricas
print("\nVariables Categóricas:")

# Distribución de Tiendas (store_id)
store_dist = all_transactions_df['store_id'].value_counts()
print("Distribución de Tiendas:\n", store_dist)

# Distribución de Categorías (de productos comprados)
category_dist = exploded_products['category_name'].value_counts()
print("\nDistribución de Categorías de Productos:\n", category_dist.head(10))  # Top 10

# Distribución de Productos (top 10 más frecuentes)
product_dist = exploded_products['product_id'].value_counts()
print("\nDistribución de Productos (Top 10):\n", product_dist.head(10))


# %%
top_categories = category_dist.head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=top_categories.values, y=top_categories.index)
plt.title('Top 10 Categorías Más Compradas')
plt.xlabel('Frecuencia')
plt.ylabel('Categoría')
plt.show()


# %%
plt.figure(figsize=(10,6))
sns.histplot(all_transactions_df['num_products'], bins=30, kde=True)
plt.title('Distribución del Número de Productos por Transacción')
plt.xlabel('Número de Productos')
plt.ylabel('Frecuencia')
plt.show()

# %%
transactions_per_day = all_transactions_df.groupby(all_transactions_df['date'].dt.date).size()

plt.figure(figsize=(12,6))
transactions_per_day.plot()
plt.title('Número de Transacciones por Día')
plt.xlabel('Fecha')
plt.ylabel('Cantidad de Transacciones')
plt.grid(True)
plt.show()


# %%
transactions_per_month = all_transactions_df.groupby(all_transactions_df['date'].dt.to_period('M')).size()
transactions_per_month.plot(kind='bar', figsize=(10,5), title='Transacciones por Mes')
plt.xlabel('Mes')
plt.ylabel('Número de Transacciones')
plt.show()


# %%
avg_products_over_time = all_transactions_df.groupby(all_transactions_df['date'].dt.to_period('M'))['num_products'].mean()

plt.figure(figsize=(10,6))
avg_products_over_time.plot(marker='o')
plt.title('Promedio de Productos por Transacción a lo Largo del Tiempo')
plt.xlabel('Mes')
plt.ylabel('Promedio de Productos')
plt.grid(True)
plt.show()


# %% [markdown]
# ## Análisis 

# %% [markdown]
# Análisis temporal
# - Ventas diarias, semanales y mensuales.
# - Picos de ventas por día de la semana y hora del día.
# - Tendencias y estacionalidad.
#
# Análisis por cliente
# - Frecuencia de compra.
# - Tiempo promedio entre compras.
# - Algún tipo de segmentación.
#
# Análisis por producto
# - Productos más vendidos.
# - Investigar reglas de asociación

# %%
transacciones= all_transactions_df.copy()

# %% [markdown]
# ## Análisis temporal

# %% [markdown]
# ### Ventas diarias, semanales y mensuales

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Convertir fecha a datetime si no está en ese formato
transacciones['date'] = pd.to_datetime(transacciones['date'])

# Ventas diarias
ventas_diarias = transacciones.groupby(transacciones['date'].dt.date).agg({
    'num_products': 'sum',
    'customer_id': 'nunique'
}).rename(columns={'num_products': 'total_ventas', 'customer_id': 'clientes_unicos'})

# Ventas semanales
ventas_semanales = transacciones.groupby(transacciones['date'].dt.to_period('W')).agg({
    'num_products': 'sum',
    'customer_id': 'nunique'
}).rename(columns={'num_products': 'total_ventas', 'customer_id': 'clientes_unicos'})

# Ventas mensuales
ventas_mensuales = transacciones.groupby(transacciones['date'].dt.to_period('M')).agg({
    'num_products': 'sum',
    'customer_id': 'nunique'
}).rename(columns={'num_products': 'total_ventas', 'customer_id': 'clientes_unicos'})

# Visualización
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Diario
axes[0].plot(ventas_diarias.index, ventas_diarias['total_ventas'])
axes[0].set_title('Ventas Diarias')
axes[0].set_xlabel('Fecha')
axes[0].set_ylabel('Total Ventas')
axes[0].tick_params(axis='x', rotation=45)

# Semanal
axes[1].bar(range(len(ventas_semanales)), ventas_semanales['total_ventas'])
axes[1].set_title('Ventas Semanales')
axes[1].set_xlabel('Semanas')
axes[1].set_ylabel('Total Ventas')

# Mensual
axes[2].bar(ventas_mensuales.index.astype(str), ventas_mensuales['total_ventas'])
axes[2].set_title('Ventas Mensuales')
axes[2].set_xlabel('Mes')
axes[2].set_ylabel('Total Ventas')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Picos de ventas por día de la semana y hora del día.

# %%
# Extraer día de la semana y hora
transacciones['dia_semana'] = transacciones['date'].dt.day_name()
transacciones['hora_dia'] = transacciones['date'].dt.hour

# Ventas por día de la semana
ventas_dia_semana = transacciones.groupby('dia_semana').agg({
    'num_products': 'sum',
    'customer_id': 'nunique'
}).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Ventas por hora del día
ventas_hora_dia = transacciones.groupby('hora_dia').agg({
    'num_products': 'sum',
    'customer_id': 'nunique'
})

# Visualización
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Ventas por día de la semana
axes[0,0].bar(ventas_dia_semana.index, ventas_dia_semana['num_products'])
axes[0,0].set_title('Ventas por Día de la Semana')
axes[0,0].set_xlabel('Día de la Semana')
axes[0,0].set_ylabel('Total Ventas')
axes[0,0].tick_params(axis='x', rotation=45)

# Clientes únicos por día de la semana
axes[0,1].bar(ventas_dia_semana.index, ventas_dia_semana['customer_id'])
axes[0,1].set_title('Clientes Únicos por Día de la Semana')
axes[0,1].set_xlabel('Día de la Semana')
axes[0,1].set_ylabel('Clientes Únicos')
axes[0,1].tick_params(axis='x', rotation=45)

# Ventas por hora del día
axes[1,0].plot(ventas_hora_dia.index, ventas_hora_dia['num_products'], marker='o')
axes[1,0].set_title('Ventas por Hora del Día')
axes[1,0].set_xlabel('Hora del Día')
axes[1,0].set_ylabel('Total Ventas')

# Clientes únicos por hora del día
axes[1,1].plot(ventas_hora_dia.index, ventas_hora_dia['customer_id'], marker='o')
axes[1,1].set_title('Clientes Únicos por Hora del Día')
axes[1,1].set_xlabel('Hora del Día')
axes[1,1].set_ylabel('Clientes Únicos')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Tendencias y estacionalidad.

# %%

from statsmodels.graphics.tsaplots import plot_acf
# Preparar datos para análisis de series temporales
serie_temporal = transacciones.set_index('date')['num_products'].resample('D').sum()

print(f"Rango de fechas: {transacciones['date'].min()} a {transacciones['date'].max()}")
print(f"Número de meses de datos: {serie_temporal.resample('M').count().shape[0]}")

# Análisis alternativo para datos limitados
def analizar_tendencias_estacionalidad(serie_temporal):
    # Convertir a DataFrame para facilitar el análisis
    df_temporal = serie_temporal.reset_index()
    df_temporal.columns = ['fecha', 'ventas']

    # Extraer componentes de fecha
    df_temporal['mes'] = df_temporal['fecha'].dt.month
    df_temporal['dia_semana'] = df_temporal['fecha'].dt.day_name()
    df_temporal['semana'] = df_temporal['fecha'].dt.isocalendar().week

    # 1. Análisis de tendencia con rolling average
    df_temporal['media_movil_7d'] = df_temporal['ventas'].rolling(window=7, min_periods=1).mean()
    df_temporal['media_movil_30d'] = df_temporal['ventas'].rolling(window=30, min_periods=1).mean()

    # 2. Análisis de estacionalidad semanal
    estacionalidad_semanal = df_temporal.groupby('dia_semana')['ventas'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])

    # 3. Análisis de estacionalidad mensual (si hay datos suficientes)
    if df_temporal['mes'].nunique() >= 2:
        estacionalidad_mensual = df_temporal.groupby('mes')['ventas'].mean()
    else:
        estacionalidad_mensual = None

    return df_temporal, estacionalidad_semanal, estacionalidad_mensual

# Aplicar análisis
df_analisis, estacionalidad_semanal, estacionalidad_mensual = analizar_tendencias_estacionalidad(serie_temporal)

# Visualización de tendencias y estacionalidad
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Serie temporal con medias móviles
axes[0, 0].plot(df_analisis['fecha'], df_analisis['ventas'], alpha=0.5, label='Ventas Diarias')
axes[0, 0].plot(df_analisis['fecha'], df_analisis['media_movil_7d'], label='Media Móvil 7 días', linewidth=2)
axes[0, 0].plot(df_analisis['fecha'], df_analisis['media_movil_30d'], label='Media Móvil 30 días', linewidth=2)
axes[0, 0].set_title('Tendencia de Ventas con Medias Móviles')
axes[0, 0].set_xlabel('Fecha')
axes[0, 0].set_ylabel('Ventas')
axes[0, 0].legend()
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Estacionalidad semanal
axes[0, 1].bar(estacionalidad_semanal.index, estacionalidad_semanal.values)
axes[0, 1].set_title('Estacionalidad Semanal (Promedio por Día)')
axes[0, 1].set_xlabel('Día de la Semana')
axes[0, 1].set_ylabel('Ventas Promedio')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Análisis de crecimiento mensual
ventas_mensuales = serie_temporal.resample('M').sum()
if len(ventas_mensuales) > 1:
    crecimiento_mensual = ventas_mensuales.pct_change() * 100
    axes[1, 0].bar(ventas_mensuales.index.strftime('%Y-%m'), ventas_mensuales.values)
    axes[1, 0].set_title('Ventas Mensuales')
    axes[1, 0].set_xlabel('Mes')
    axes[1, 0].set_ylabel('Ventas Totales')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Crecimiento mensual
    axes[1, 1].plot(crecimiento_mensual.index.strftime('%Y-%m'), crecimiento_mensual.values, marker='o', color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Crecimiento Mensual (%)')
    axes[1, 1].set_xlabel('Mes')
    axes[1, 1].set_ylabel('Crecimiento %')
    axes[1, 1].tick_params(axis='x', rotation=45)
else:
    axes[1, 0].text(0.5, 0.5, 'Datos insuficientes\npara análisis mensual',
                   ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 1].text(0.5, 0.5, 'Datos insuficientes\npara análisis de crecimiento',
                   ha='center', va='center', transform=axes[1, 1].transAxes)

plt.tight_layout()
plt.show()

# Análisis de autocorrelación para patrones
plt.figure(figsize=(12, 8))

# Autocorrelación para 15 días (2 semanas)
plot_acf(serie_temporal.dropna(), lags=min(30, len(serie_temporal)-1), ax=plt.gca())
plt.title('Autocorrelación de Ventas Diarias')
plt.show()


# %%
# Métricas de tendencia detalladas
def calcular_metricas_tendencia(df_analisis):
    metricas = {}

    # Tendencia lineal simple
    x = np.arange(len(df_analisis))
    y = df_analisis['ventas'].values
    slope, intercept = np.polyfit(x, y, 1)
    metricas['pendiente_tendencia'] = slope
    metricas['tendencia_diaria'] = slope / df_analisis['ventas'].mean() * 100  # % cambio diario

    # Variabilidad
    metricas['coef_variacion'] = df_analisis['ventas'].std() / df_analisis['ventas'].mean()
    metricas['ratio_pico_valle'] = df_analisis['ventas'].max() / df_analisis['ventas'].min()

    # Estacionalidad semanal
    variacion_semanal = estacionalidad_semanal.std() / estacionalidad_semanal.mean()
    metricas['fuerza_estacionalidad_semanal'] = variacion_semanal

    return metricas

metricas_tendencia = calcular_metricas_tendencia(df_analisis)

print("=== MÉTRICAS DE TENDENCIA Y ESTACIONALIDAD ===")
print(f"Pendiente de tendencia: {metricas_tendencia['pendiente_tendencia']:.2f} ventas/día")
print(f"Tendencia diaria: {metricas_tendencia['tendencia_diaria']:.2f}%")
print(f"Coeficiente de variación: {metricas_tendencia['coef_variacion']:.2f}")
print(f"Ratio pico/valle: {metricas_tendencia['ratio_pico_valle']:.2f}")
print(f"Fuerza estacionalidad semanal: {metricas_tendencia['fuerza_estacionalidad_semanal']:.2f}")

# Análisis de patrones semanales detallados
print("\n=== ANÁLISIS ESTACIONALIDAD SEMANAL ===")
for dia, ventas in estacionalidad_semanal.items():
    variacion_vs_promedio = (ventas / estacionalidad_semanal.mean() - 1) * 100
    print(f"{dia}: {ventas:.0f} ventas promedio ({variacion_vs_promedio:+.1f}% vs promedio)")

# Identificación de outliers y patrones especiales
Q1 = df_analisis['ventas'].quantile(0.25)
Q3 = df_analisis['ventas'].quantile(0.75)
IQR = Q3 - Q1
limite_superior = Q3 + 1.5 * IQR
limite_inferior = Q1 - 1.5 * IQR

outliers = df_analisis[(df_analisis['ventas'] > limite_superior) | (df_analisis['ventas'] < limite_inferior)]

print(f"\n=== DÍAS ATÍPICOS DETECTADOS ===")
print(f"Límites: {limite_inferior:.0f} - {limite_superior:.0f} ventas")
if not outliers.empty:
    for _, outlier in outliers.iterrows():
        print(f"{outlier['fecha'].strftime('%Y-%m-%d')}: {outlier['ventas']:.0f} ventas ({outlier['dia_semana']})")
else:
    print("No se detectaron días atípicos significativos")

# %%
# Heatmap de ventas por día de semana y semana del año
if 'semana' in df_analisis.columns and df_analisis['semana'].nunique() > 1:
    # Crear heatmap semanal
    heatmap_data = df_analisis.pivot_table(
        values='ventas',
        index='semana',
        columns='dia_semana',
        aggfunc='mean'
    ).reindex(columns=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.0f', linewidths=0.5)
    plt.title('Heatmap de Ventas: Semana vs Día de la Semana')
    plt.xlabel('Día de la Semana')
    plt.ylabel('Semana del Año')
    plt.tight_layout()
    plt.show()

# Análisis de acumulado y tendencia
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Ventas acumuladas
ventas_acumuladas = df_analisis['ventas'].cumsum()
axes[0].plot(df_analisis['fecha'], ventas_acumuladas)
axes[0].set_title('Ventas Acumuladas')
axes[0].set_xlabel('Fecha')
axes[0].set_ylabel('Ventas Acumuladas')
axes[0].tick_params(axis='x', rotation=45)

# Distribución de ventas diarias
axes[1].hist(df_analisis['ventas'], bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(df_analisis['ventas'].mean(), color='red', linestyle='--', label=f'Media: {df_analisis["ventas"].mean():.0f}')
axes[1].axvline(df_analisis['ventas'].median(), color='green', linestyle='--', label=f'Mediana: {df_analisis["ventas"].median():.0f}')
axes[1].set_title('Distribución de Ventas Diarias')
axes[1].set_xlabel('Ventas Diarias')
axes[1].set_ylabel('Frecuencia')
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Análisis por cliente

# %% [markdown]
# ### Frecuencia de compra.

# %%
# Análisis de comportamiento del cliente
comportamiento_cliente = transacciones.groupby('customer_id').agg({
    'date': ['count', 'min', 'max'],
    'num_products': 'sum',
    'store_id': 'nunique'
}).round(2)

comportamiento_cliente.columns = ['total_compras', 'primera_compra', 'ultima_compra', 'total_productos', 'tiendas_visitadas']

# Calcular métricas de frecuencia
comportamiento_cliente['dias_activo'] = (comportamiento_cliente['ultima_compra'] - comportamiento_cliente['primera_compra']).dt.days
comportamiento_cliente['frecuencia_promedio'] = comportamiento_cliente['dias_activo'] / comportamiento_cliente['total_compras']

# Análisis de recencia (última compra)
ultima_fecha = transacciones['date'].max()
comportamiento_cliente['recencia'] = (ultima_fecha - comportamiento_cliente['ultima_compra']).dt.days

# Visualización del análisis de clientes
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Distribución de frecuencia de compra
axes[0,0].hist(comportamiento_cliente['frecuencia_promedio'].dropna(), bins=50, edgecolor='black')
axes[0,0].set_title('Distribución de Frecuencia de Compra')
axes[0,0].set_xlabel('Días entre Compras')
axes[0,0].set_ylabel('Número de Clientes')

# Distribución de número de compras
axes[0,1].hist(comportamiento_cliente['total_compras'], bins=50, edgecolor='black')
axes[0,1].set_title('Distribución de Número de Compras por Cliente')
axes[0,1].set_xlabel('Número de Compras')
axes[0,1].set_ylabel('Número de Clientes')

# Recencia vs Frecuencia
axes[1,0].scatter(comportamiento_cliente['frecuencia_promedio'], comportamiento_cliente['recencia'], alpha=0.5)
axes[1,0].set_title('Recencia vs Frecuencia')
axes[1,0].set_xlabel('Frecuencia Promedio (días)')
axes[1,0].set_ylabel('Recencia (días)')

# Distribución de productos por cliente
axes[1,1].hist(comportamiento_cliente['total_productos'], bins=50, edgecolor='black')
axes[1,1].set_title('Distribución de Productos por Cliente')
axes[1,1].set_xlabel('Total Productos Comprados')
axes[1,1].set_ylabel('Número de Clientes')

# Tiendas visitadas por cliente
axes[0,2].hist(comportamiento_cliente['tiendas_visitadas'], bins=20, edgecolor='black')
axes[0,2].set_title('Tiendas Visitadas por Cliente')
axes[0,2].set_xlabel('Número de Tiendas')
axes[0,2].set_ylabel('Número de Clientes')

plt.tight_layout()
plt.show()

# Métricas resumen del análisis de clientes
print("=== RESUMEN ANÁLISIS DE CLIENTES ===")
print(f"Total clientes únicos: {len(comportamiento_cliente)}")
print(f"Compras promedio por cliente: {comportamiento_cliente['total_compras'].mean():.2f}")
print(f"Frecuencia promedio de compra: {comportamiento_cliente['frecuencia_promedio'].mean():.2f} días")
print(f"Productos promedio por cliente: {comportamiento_cliente['total_productos'].mean():.2f}")
print(f"Recencia promedio: {comportamiento_cliente['recencia'].mean():.2f} días")

# %% [markdown]
# ### Tiempo promedio entre compras

# %%
all_transactions_df = all_transactions_df.sort_values(by=['customer_id', 'date'])
all_transactions_df['prev_date'] = all_transactions_df.groupby('customer_id')['date'].shift()
all_transactions_df['days_between'] = (all_transactions_df['date'] - all_transactions_df['prev_date']).dt.days

avg_purchase_interval = all_transactions_df.groupby('customer_id')['days_between'].mean()

print("\nTiempo promedio entre compras (días):")
print(avg_purchase_interval.describe())


# %%
plt.figure(figsize=(8,5))
plt.hist(avg_purchase_interval.dropna(), bins=50)
plt.title("Distribución del Tiempo Promedio entre Compras (días)")
plt.xlabel("Días entre compras")
plt.ylabel("Número de clientes")
plt.show()

# %% [markdown]
# ### Segmentación

# %% [markdown]
# avg_purchase_interval representa el promedio de días que un cliente tarda entre una compra y la siguiente, indicando qué tan frecuentemente realiza compras a lo largo del tiempo.

# %% [markdown]
# **Segmentación por Frecuencia**: clasifica a los clientes según cuántas compras han realizado, desde los que compran pocas veces (“Ocasional”) hasta los que compran con mucha regularidad (“Muy Frecuente”).
#
# **Segmentación por Intervalo de Compra**: clasifica a los clientes según cada cuánto tiempo compran en promedio, desde los que compran con muy poca distancia entre compras (“Muy Frecuente”) hasta los que pasan largos periodos sin comprar (“Ocasional”).

# %%
# Calcular frecuencia de compra por cliente
customer_freq = all_transactions_df['customer_id'].value_counts().reset_index(name='freq')

# Calcular intervalo promedio entre compras
customer_segments = pd.DataFrame({
    "freq": customer_freq.set_index('customer_id')['freq'],
    "avg_purchase_interval": all_transactions_df.groupby('customer_id')['days_between'].mean()
})

# Segmentación por frecuencia
customer_segments['freq_segment'] = pd.cut(
    customer_segments['freq'],
    bins=[0, 2, 5, 10, 1000],
    labels=["Ocasional", "Moderado", "Frecuente", "Muy Frecuente"]
)

# Segmentación por intervalo de compra (valores altos = menos frecuente)
customer_segments['interval_segment'] = pd.cut(
    customer_segments['avg_purchase_interval'],
    bins=[0, 5, 15, 30, 1000],
    labels=["Muy Frecuente", "Frecuente", "Moderado", "Ocasional"]
)

print("\nSegmentación de Clientes (Frecuencia):")
print(customer_segments['freq_segment'].value_counts())

print("\nSegmentación de Clientes (Intervalo de Compra):")
print(customer_segments['interval_segment'].value_counts())


# %%
import matplotlib.pyplot as plt

# === Gráfico 1: Segmentación por Frecuencia ===
plt.figure(figsize=(8,6))
customer_segments['freq_segment'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Distribución de Segmentos por Frecuencia de Compra", fontsize=14)
plt.xlabel("Segmento de Frecuencia", fontsize=12)
plt.ylabel("Número de Clientes", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# === Gráfico 2: Segmentación por Intervalo de Compra ===
plt.figure(figsize=(8,6))
customer_segments['interval_segment'].value_counts().sort_index().plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title("Distribución de Segmentos por Intervalo Promedio de Compra", fontsize=14)
plt.xlabel("Segmento por Intervalo de Compra", fontsize=12)
plt.ylabel("Número de Clientes", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Análisis por producto

# %% [markdown]
# ### Productos más vendidos



# %% [markdown]
# ### Reglas de asociación


# %% [markdown]
# ### Recomendador de Productos

# %% [markdown]
# Desarrollar un modelo básico de recomendación usando técnicas de filtrado colaborativo o
# reglas de asociación:
#

# %%
import pandas as pd
rules = pd.read_pickle("association_rules.pkl")

# %% [markdown]
# **Dado un cliente**: sugerir productos complementarios o similares a los que ha
# comprado.

# %%
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

cliente_producto = exploded_products.groupby(['customer_id', 'product_id']).size().unstack(fill_value=0)
cliente_producto = (cliente_producto > 0).astype(int)


def recomendar_por_cliente(cliente_id, top_n=5):
    if cliente_id not in cliente_producto.index:
        return f"Cliente {cliente_id} no encontrado."
    
    cliente_vector = cliente_producto.loc[[cliente_id]]
    
    # Calcular similitud solo contra todos los demás
    similitudes = cosine_similarity(cliente_vector, cliente_producto)[0]
    sim_series = pd.Series(similitudes, index=cliente_producto.index).sort_values(ascending=False)[1:50]
    
    # Clientes similares
    clientes_similares = sim_series.index
    
    # Productos del cliente actual
    productos_cliente = set(cliente_producto.loc[cliente_id][cliente_producto.loc[cliente_id] > 0].index)
    
    # Promedio de compras entre clientes similares
    productos_similares = cliente_producto.loc[clientes_similares]
    productos_prom = productos_similares.mean().sort_values(ascending=False)
    
    # Filtrar productos que el cliente no tiene
    recomendaciones = productos_prom[~productos_prom.index.isin(productos_cliente)].head(top_n)
    
    return list(recomendaciones.index)



# %%
def mapear_categorias_productos(product_ids):
    """
    Recibe una lista/Serie de product_id y devuelve sus categorías asociadas.
    """
    mapeo = product_category_df.merge(categories_df, left_on='v.code', right_on='category_id', how='left')
    mapeo = mapeo[['v.Code_pr', 'category_name']].drop_duplicates()
    
    return pd.Series(product_ids).map(dict(zip(mapeo['v.Code_pr'], mapeo['category_name'])))



# %%
recomendar_por_cliente(1508)

# %%
mapear_categorias_productos(recomendar_por_cliente(1508))


# %% [markdown]
# **Dado un producto**: recomendar otros productos que suelen comprarse junto a él.
#

# %%
def recomendar_por_producto(producto_id, top_n=5):
    """
    Retorna los productos que suelen comprarse junto al producto dado.
    """
    recomendaciones = rules[rules['antecedents'].apply(lambda x: producto_id in x)]
    if recomendaciones.empty:
        return f"No hay reglas disponibles para el producto {producto_id}."
    
    recomendaciones = recomendaciones.sort_values(by='lift', ascending=False).head(top_n)
    return recomendaciones[['antecedents', 'consequents', 'confidence', 'lift']]


# %%
recomendar_por_producto(20)


# %%
def recomendar_por_producto_unico_antecedente(producto_id, top_n=5):
 
    mask = rules['antecedents'].apply(lambda x: x == frozenset([producto_id]))
    
    recomendaciones = rules[mask]
    
    if recomendaciones.empty:
        return f"No hay reglas disponibles para el producto {producto_id}."
    
    recomendaciones = recomendaciones.sort_values(by='lift', ascending=False).head(top_n)
    return recomendaciones[['antecedents', 'consequents', 'confidence', 'lift']]



# %%
recomendar_por_producto_unico_antecedente(20)
