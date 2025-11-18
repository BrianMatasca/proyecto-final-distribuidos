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

# %% [markdown]
# # Análisis y Modelado Analítico de Transacciones de Supermercado

# %%
import mercury as mr
import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

# %%
# Configuración de la app Mercury
app = mr.App(
    title="Análisis de Transacciones de Supermercado",
    description="Dashboard interactivo para análisis descriptivo y diagnóstico de transacciones",
    show_code=False,
    show_prompt=False,
    continuous_update=True,
    static_notebook=False,
    show_sidebar=True,
    full_screen=True,
    allow_download=True,
    allow_share=True,
    stop_on_error=False
)

# %% [markdown]
# ## 1. Carga y Preparación de Datos

# %%
PRODUCTS_FOLDER = 'Products/'
TRANSACTIONS_FOLDER = 'Transactions/'

# Cargar categorías
categories_df = pd.read_csv(os.path.join(PRODUCTS_FOLDER, 'Categories.csv'), 
                           sep='|', header=None, names=['category_id', 'category_name'])

# Cargar relación producto-categoría
product_category_df = pd.read_csv(os.path.join(PRODUCTS_FOLDER, 'ProductCategory.csv'), sep='|')

# Cargar transacciones
transaction_files = glob(os.path.join(TRANSACTIONS_FOLDER, '*_Tran.csv'))
transactions_dfs = []

for file in transaction_files:
    df = pd.read_csv(file, sep='|', header=None, 
                    names=['date', 'store_id', 'customer_id', 'products'])
    df['products'] = df['products'].str.split(' ')
    transactions_dfs.append(df)

all_transactions_df = pd.concat(transactions_dfs, ignore_index=True)
all_transactions_df['products'] = all_transactions_df['products'].apply(tuple)
all_transactions_df['date'] = pd.to_datetime(all_transactions_df['date'])
all_transactions_df['num_products'] = all_transactions_df['products'].apply(len)

# Expandir productos
exploded_products = all_transactions_df.explode('products')
exploded_products['product_id'] = pd.to_numeric(exploded_products['products'], errors='coerce')
exploded_products = exploded_products.merge(product_category_df, 
                                           left_on='product_id', 
                                           right_on='v.Code_pr', how='left')
exploded_products.rename(columns={'v.code': 'category_id'}, inplace=True)
exploded_products = exploded_products.merge(categories_df, on='category_id', how='left')

mr.Md("### ✅ Datos cargados exitosamente")

# %% [markdown]
# ## 2. Resumen Ejecutivo

# %%
mr.Md("## 📊 Resumen Ejecutivo")

# Calcular métricas clave
total_ventas = all_transactions_df['num_products'].sum()
num_transacciones = len(all_transactions_df)
num_clientes = all_transactions_df['customer_id'].nunique()
num_productos = exploded_products['product_id'].nunique()

# %%
# Mostrar métricas principales con NumberBox
mr.NumberBox(
    data=int(total_ventas),
    title="Total de Unidades Vendidas",
    background_color="#e3f2fd"
)

mr.NumberBox(
    data=num_transacciones,
    title="Número de Transacciones",
    background_color="#f3e5f5"
)

mr.NumberBox(
    data=num_clientes,
    title="Clientes Únicos",
    background_color="#e8f5e9"
)

mr.NumberBox(
    data=num_productos,
    title="Productos Únicos",
    background_color="#fff3e0"
)

# %% [markdown]
# ### Top 10 Productos Más Vendidos

# %%
top_productos = exploded_products['product_id'].value_counts().head(10)

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_productos)), top_productos.values, color='steelblue')
plt.yticks(range(len(top_productos)), [f"Producto {int(p)}" for p in top_productos.index])
plt.xlabel('Frecuencia de Compra')
plt.title('Top 10 Productos Más Comprados')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Top 10 Clientes

# %%
top_clientes = all_transactions_df['customer_id'].value_counts().head(10)

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_clientes)), top_clientes.values, color='coral')
plt.yticks(range(len(top_clientes)), [f"Cliente {int(c)}" for c in top_clientes.index])
plt.xlabel('Número de Compras')
plt.title('Top 10 Clientes con Mayor Número de Compras')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Categorías Más Populares

# %%
category_dist = exploded_products['category_name'].value_counts().head(10)

plt.figure(figsize=(10, 6))
plt.pie(category_dist.values, labels=category_dist.index, autopct='%1.1f%%', startangle=90)
plt.title('Top 10 Categorías Más Compradas')
plt.axis('equal')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Visualizaciones Analíticas

# %%
mr.Md("## 📈 Visualizaciones Analíticas")

# %% [markdown]
# ### Serie de Tiempo - Ventas por Día

# %%
transactions_per_day = all_transactions_df.groupby(all_transactions_df['date'].dt.date).size()

plt.figure(figsize=(14, 6))
transactions_per_day.plot(color='darkblue', linewidth=1.5)
plt.title('Número de Transacciones por Día')
plt.xlabel('Fecha')
plt.ylabel('Cantidad de Transacciones')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Boxplot - Distribución de Productos por Cliente

# %%
cliente_productos = all_transactions_df.groupby('customer_id')['num_products'].sum()

plt.figure(figsize=(10, 6))
sns.boxplot(x=cliente_productos, color='lightgreen')
plt.title('Distribución de Total de Productos por Cliente')
plt.xlabel('Total de Productos Comprados')
plt.tight_layout()
plt.show()

mr.Md(f"""
**Estadísticas:**
- Media: {cliente_productos.mean():.2f}
- Mediana: {cliente_productos.median():.2f}
- Desviación Estándar: {cliente_productos.std():.2f}
""")

# %% [markdown]
# ### Heatmap - Correlación entre Variables

# %%
# Crear matriz de correlación
customer_features = all_transactions_df.groupby('customer_id').agg({
    'num_products': ['sum', 'mean'],
    'date': 'count',
    'store_id': 'nunique'
})
customer_features.columns = ['total_productos', 'promedio_productos', 'frecuencia', 'tiendas_visitadas']

correlation_matrix = customer_features.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlación entre Variables de Comportamiento del Cliente')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Análisis Temporal

# %%
mr.Md("## 📅 Análisis Temporal")

# Extraer características temporales
all_transactions_df['dia_semana'] = all_transactions_df['date'].dt.day_name()
all_transactions_df['hora_dia'] = all_transactions_df['date'].dt.hour

# %% [markdown]
# ### Ventas por Día de la Semana

# %%
ventas_dia_semana = all_transactions_df.groupby('dia_semana')['num_products'].sum().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)

plt.figure(figsize=(12, 6))
plt.bar(ventas_dia_semana.index, ventas_dia_semana.values, color='teal')
plt.title('Ventas por Día de la Semana')
plt.xlabel('Día de la Semana')
plt.ylabel('Total Ventas')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Ventas por Hora del Día

# %%
ventas_hora = all_transactions_df.groupby('hora_dia')['num_products'].sum()

plt.figure(figsize=(12, 6))
plt.plot(ventas_hora.index, ventas_hora.values, marker='o', linewidth=2, color='darkred')
plt.title('Ventas por Hora del Día')
plt.xlabel('Hora del Día')
plt.ylabel('Total Ventas')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Tendencias y Estacionalidad

# %%
serie_temporal = all_transactions_df.set_index('date')['num_products'].resample('D').sum()

df_temporal = serie_temporal.reset_index()
df_temporal.columns = ['fecha', 'ventas']
df_temporal['media_movil_7d'] = df_temporal['ventas'].rolling(window=7, min_periods=1).mean()
df_temporal['media_movil_30d'] = df_temporal['ventas'].rolling(window=30, min_periods=1).mean()

plt.figure(figsize=(14, 6))
plt.plot(df_temporal['fecha'], df_temporal['ventas'], alpha=0.3, label='Ventas Diarias')
plt.plot(df_temporal['fecha'], df_temporal['media_movil_7d'], label='Media Móvil 7 días', linewidth=2)
plt.plot(df_temporal['fecha'], df_temporal['media_movil_30d'], label='Media Móvil 30 días', linewidth=2)
plt.title('Tendencia de Ventas con Medias Móviles')
plt.xlabel('Fecha')
plt.ylabel('Ventas')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Análisis por Cliente

# %%
mr.Md("## 👥 Análisis por Cliente")

# Calcular comportamiento del cliente
all_transactions_df = all_transactions_df.sort_values(by=['customer_id', 'date'])
all_transactions_df['prev_date'] = all_transactions_df.groupby('customer_id')['date'].shift()
all_transactions_df['days_between'] = (all_transactions_df['date'] - all_transactions_df['prev_date']).dt.days

comportamiento_cliente = all_transactions_df.groupby('customer_id').agg({
    'date': ['count', 'min', 'max'],
    'num_products': 'sum',
    'store_id': 'nunique',
    'days_between': 'mean'
})

comportamiento_cliente.columns = ['total_compras', 'primera_compra', 'ultima_compra', 
                                  'total_productos', 'tiendas_visitadas', 'dias_entre_compras']

# %% [markdown]
# ### Distribución de Frecuencia de Compra

# %%
plt.figure(figsize=(12, 6))
plt.hist(comportamiento_cliente['total_compras'], bins=50, edgecolor='black', color='skyblue')
plt.title('Distribución de Número de Compras por Cliente')
plt.xlabel('Número de Compras')
plt.ylabel('Número de Clientes')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

mr.Md(f"""
**Estadísticas de Frecuencia:**
- Compras promedio por cliente: {comportamiento_cliente['total_compras'].mean():.2f}
- Mediana de compras: {comportamiento_cliente['total_compras'].median():.2f}
""")

# %% [markdown]
# ### Tiempo Promedio entre Compras

# %%
plt.figure(figsize=(12, 6))
plt.hist(comportamiento_cliente['dias_entre_compras'].dropna(), bins=50, 
         edgecolor='black', color='lightcoral')
plt.title('Distribución del Tiempo Promedio entre Compras')
plt.xlabel('Días entre Compras')
plt.ylabel('Número de Clientes')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

mr.Md(f"""
**Estadísticas de Intervalo:**
- Intervalo promedio: {comportamiento_cliente['dias_entre_compras'].mean():.2f} días
- Mediana: {comportamiento_cliente['dias_entre_compras'].median():.2f} días
""")

# %% [markdown]
# ### Segmentación de Clientes

# %%
customer_freq = all_transactions_df['customer_id'].value_counts().reset_index(name='freq')

customer_segments = pd.DataFrame({
    "freq": customer_freq.set_index('customer_id')['freq'],
    "avg_purchase_interval": all_transactions_df.groupby('customer_id')['days_between'].mean()
})

customer_segments['freq_segment'] = pd.cut(
    customer_segments['freq'],
    bins=[0, 2, 5, 10, 1000],
    labels=["Ocasional", "Moderado", "Frecuente", "Muy Frecuente"]
)

customer_segments['interval_segment'] = pd.cut(
    customer_segments['avg_purchase_interval'],
    bins=[0, 5, 15, 30, 1000],
    labels=["Muy Frecuente", "Frecuente", "Moderado", "Ocasional"]
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

customer_segments['freq_segment'].value_counts().sort_index().plot(
    kind='bar', color='steelblue', edgecolor='black', ax=axes[0]
)
axes[0].set_title('Segmentación por Frecuencia de Compra')
axes[0].set_xlabel('Segmento')
axes[0].set_ylabel('Número de Clientes')
axes[0].tick_params(axis='x', rotation=0)
axes[0].grid(axis='y', alpha=0.3)

customer_segments['interval_segment'].value_counts().sort_index().plot(
    kind='bar', color='lightgreen', edgecolor='black', ax=axes[1]
)
axes[1].set_title('Segmentación por Intervalo de Compra')
axes[1].set_xlabel('Segmento')
axes[1].set_ylabel('Número de Clientes')
axes[1].tick_params(axis='x', rotation=0)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Análisis Avanzado

# %%
mr.Md("## 🔬 Análisis Avanzado")

# %% [markdown]
# ### A. Segmentación de Clientes con K-Means

# %%
mr.Md("""
### Segmentación de Clientes con K-Means

*Espacio reservado para implementación de clustering*

**Variables sugeridas:**
- Frecuencia de compra
- Número de productos distintos
- Volumen total
- Diversidad de categorías
""")

# %% [markdown]
# ### B. Sistema de Recomendación de Productos

# %%
mr.Md("## 🎯 Sistema de Recomendación de Productos")

# Cargar reglas de asociación
try:
    rules = pd.read_pickle("association_rules.pkl")
    mr.Md("✅ **Reglas de asociación cargadas correctamente**")
except FileNotFoundError:
    mr.Md("⚠️ **Archivo 'association_rules.pkl' no encontrado**")
    rules = None

# %%
# Widget para seleccionar tipo de recomendación
tipo_recomendacion = mr.Select(
    value="cliente",
    choices=["cliente", "producto"],
    label="Tipo de Recomendación"
)

# Widgets condicionales - definir ambos pero con valores por defecto
if tipo_recomendacion.value == "cliente":
    cliente_id = mr.Numeric(
        value=530,
        label="ID del Cliente",
        min=1,
        max=999999,
        step=1
    )
    producto_id = 1007  # Valor por defecto cuando no se usa
    
else:
    producto_id = mr.Numeric(
        value=1007,
        label="ID del Producto",
        min=1,
        max=999999,
        step=1
    )
    cliente_id = 530  # Valor por defecto cuando no se usa

# Número de recomendaciones (común para ambos)
num_recomendaciones = mr.Slider(
    value=5,
    min=1,
    max=20,
    step=1,
    label="Número de Recomendaciones"
)

# %%
# Mostrar título según tipo de recomendación
if tipo_recomendacion == "cliente":
    mr.Md(f"### Recomendaciones para Cliente {cliente_id}")
else:
    mr.Md(f"### Recomendaciones para Producto {producto_id}")

# %% [markdown]
# #### Funciones de Recomendación

# %%
def mapear_categorias_productos(product_ids):
    """Mapea IDs de productos a sus categorías"""
    mapeo = product_category_df.merge(categories_df, left_on='v.code', 
                                     right_on='category_id', how='left')
    mapeo = mapeo[['v.Code_pr', 'category_name']].drop_duplicates()
    return pd.Series(product_ids).map(dict(zip(mapeo['v.Code_pr'], mapeo['category_name'])))

def recomendar_por_cliente(cliente_id, top_n=5):
    """Recomienda productos para un cliente usando filtrado colaborativo"""
    cliente_producto = exploded_products.groupby(['customer_id', 'product_id']).size().unstack(fill_value=0)
    cliente_producto = (cliente_producto > 0).astype(int)
    
    if cliente_id not in cliente_producto.index:
        return f"Cliente {cliente_id} no encontrado."
    
    cliente_vector = cliente_producto.loc[[cliente_id]]
    similitudes = cosine_similarity(cliente_vector, cliente_producto)[0]
    sim_series = pd.Series(similitudes, index=cliente_producto.index).sort_values(ascending=False)[1:50]
    
    clientes_similares = sim_series.index
    productos_cliente = set(cliente_producto.loc[cliente_id][cliente_producto.loc[cliente_id] > 0].index)
    
    productos_similares = cliente_producto.loc[clientes_similares]
    productos_prom = productos_similares.mean().sort_values(ascending=False)
    
    recomendaciones = productos_prom[~productos_prom.index.isin(productos_cliente)].head(top_n)
    return list(recomendaciones.index)

def recomendar_por_producto(producto_id, top_n=5):
    """Recomienda productos complementarios usando reglas de asociación"""
    if rules is None:
        return "Reglas de asociación no disponibles"
    
    mask = rules['antecedents'].apply(lambda x: x == frozenset([producto_id]))
    recomendaciones = rules[mask]
    
    if recomendaciones.empty:
        return f"No hay reglas disponibles para el producto {producto_id}."
    
    recomendaciones = recomendaciones.sort_values(by='lift', ascending=False).head(top_n)
    return recomendaciones[['antecedents', 'consequents', 'confidence', 'lift']]

# %%
# Ejecutar recomendación según tipo seleccionado
if tipo_recomendacion == "cliente":
    recomendaciones = recomendar_por_cliente(cliente_id, num_recomendaciones)
    
    if isinstance(recomendaciones, str):
        mr.Md(f"⚠️ {recomendaciones}")
    else:
        mr.Md("#### Productos Recomendados:")
        
        # Crear DataFrame con recomendaciones
        df_recom = pd.DataFrame({
            'Producto ID': recomendaciones,
            'Categoría': mapear_categorias_productos(recomendaciones)
        })
        
        mr.Table(data=df_recom, text_align="left")
        
        # Gráfico
        categorias = mapear_categorias_productos(recomendaciones).value_counts()
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(categorias)), categorias.values, color='mediumpurple')
        plt.yticks(range(len(categorias)), categorias.index)
        plt.xlabel('Frecuencia')
        plt.title(f'Categorías Recomendadas para Cliente {cliente_id}')
        plt.tight_layout()
        plt.show()

else:
    recomendaciones = recomendar_por_producto(producto_id, num_recomendaciones)
    
    if isinstance(recomendaciones, str):
        mr.Md(f"⚠️ {recomendaciones}")
    else:
        mr.Md("#### Productos que se Compran Juntos:")
        
        # Mostrar tabla de reglas
        df_display = recomendaciones.copy()
        df_display['antecedents'] = df_display['antecedents'].apply(lambda x: ', '.join(map(str, x)))
        df_display['consequents'] = df_display['consequents'].apply(lambda x: ', '.join(map(str, x)))
        
        mr.Table(data=df_display, text_align="left")
        
        # Gráfico de lift
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(recomendaciones)), recomendaciones['lift'].values, color='orange')
        plt.yticks(range(len(recomendaciones)), 
                  [f"Regla {i+1}" for i in range(len(recomendaciones))])
        plt.xlabel('Lift')
        plt.title(f'Fuerza de Asociación para Producto {producto_id}')
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 7. Conclusiones

# %%
mr.Md("""
## 📋 Conclusiones Principales

### Hallazgos Clave:

1. **Patrones Temporales:**
   - Identificación de días y horas pico de ventas
   - Tendencias estacionales en el comportamiento de compra

2. **Comportamiento del Cliente:**
   - Segmentación clara de clientes por frecuencia y recencia
   - Identificación de clientes de alto valor

3. **Análisis de Productos:**
   - Productos y categorías más populares
   - Relaciones de compra complementaria

4. **Sistema de Recomendación:**
   - Implementación de filtrado colaborativo para clientes
   - Uso de reglas de asociación para productos complementarios

### Aplicaciones Empresariales:

- **Marketing Personalizado:** Usar segmentación para campañas dirigidas
- **Gestión de Inventario:** Optimizar stock basado en patrones de venta
- **Cross-selling:** Implementar recomendaciones en punto de venta
- **Programas de Lealtad:** Diseñar incentivos para diferentes segmentos
""")

# %%
mr.Md("---")
mr.Md("**Desarrollado para Análisis de Transacciones de Supermercado** | Dashboard Interactivo con Mercury")