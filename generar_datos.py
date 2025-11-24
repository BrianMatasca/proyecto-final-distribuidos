import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Crear carpetas si no existen
os.makedirs('datos_nuevos/Products', exist_ok=True)
os.makedirs('datos_nuevos/Transactions', exist_ok=True)

# 1. Generar Categories.csv
categories_data = [
    [1, 'Electrónica'],
    [2, 'Ropa'],
    [3, 'Alimentos'],
    [4, 'Hogar'],
    [5, 'Deportes'],
    [6, 'Juguetes'],
    [7, 'Libros'],
    [8, 'Belleza'],
    [9, 'Herramientas'],
    [10, 'Jardinería']
]

categories_df = pd.DataFrame(categories_data, columns=['category_id', 'category_name'])
categories_df.to_csv('datos_nuevos/Products/Categories.csv', sep='|', header=False, index=False)
print("✓ Categories.csv generado")

# 2. Generar ProductCategory.csv
# Crear 100 productos distribuidos en las categorías
np.random.seed(42)
products = []
for product_id in range(1, 101):
    category_id = np.random.choice(categories_df['category_id'].values)
    products.append({'v.Code_pr': product_id, 'v.code': category_id})

product_category_df = pd.DataFrame(products)
product_category_df.to_csv('datos_nuevos/Products/ProductCategory.csv', sep='|', index=False)
print("✓ ProductCategory.csv generado")

# 3. Generar 105_Tran.csv
# Generar transacciones para los últimos 90 días
np.random.seed(42)
transactions = []
start_date = datetime.now() - timedelta(days=90)
store_id = 105

# Generar 500 transacciones
for _ in range(500):
    # Fecha aleatoria en los últimos 90 días
    random_days = np.random.randint(0, 90)
    transaction_date = start_date + timedelta(days=random_days)
    date_str = transaction_date.strftime('%Y-%m-%d')
    
    # Customer ID aleatorio entre 1000 y 9999
    customer_id = np.random.randint(1000, 10000)
    
    # Número de productos en la transacción (entre 1 y 10)
    num_products = np.random.randint(1, 11)
    
    # Seleccionar productos aleatorios
    product_ids = np.random.choice(range(1, 101), size=num_products, replace=False)
    products_str = ' '.join(map(str, product_ids))
    
    transactions.append([date_str, store_id, customer_id, products_str])

transactions_df = pd.DataFrame(transactions, 
                               columns=['date', 'store_id', 'customer_id', 'products'])
transactions_df = transactions_df.sort_values('date')
transactions_df.to_csv('datos_nuevos/Transactions/105_Tran.csv', sep='|', header=False, index=False)
print("✓ 105_Tran.csv generado")

# Mostrar estadísticas
print("\n=== ESTADÍSTICAS ===")
print(f"Categorías: {len(categories_df)}")
print(f"Productos: {len(product_category_df)}")
print(f"Transacciones: {len(transactions_df)}")
print(f"\nRango de fechas: {transactions_df['date'].min()} a {transactions_df['date'].max()}")
print(f"\nMuestra de Categories.csv:")
print(categories_df.head())
print(f"\nMuestra de ProductCategory.csv:")
print(product_category_df.head())
print(f"\nMuestra de 105_Tran.csv:")
print(transactions_df.head())