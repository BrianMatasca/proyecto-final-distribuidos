# generator_synthetic_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generar_datos_sinteticos(n_registros, categories_df, product_category_df, all_transactions_df):
    """
    Genera registros sintéticos con la misma estructura que exploded_products
    
    Parámetros:
    -----------
    n_registros : int
        Número de registros a generar
    categories_df : DataFrame
        DataFrame con las categorías
    product_category_df : DataFrame 
        DataFrame con la relación producto-categoría
    all_transactions_df : DataFrame
        DataFrame original de transacciones para referencia
    
    Retorna:
    --------
    DataFrame con n_registros sintéticos con la estructura de exploded_products
    """
    
    # Obtener rangos y valores únicos de los datos reales para mantener consistencia
    fechas_min = all_transactions_df['date'].min()
    fechas_max = all_transactions_df['date'].max()
    
    store_ids = all_transactions_df['store_id'].unique()
    customer_ids_real = all_transactions_df['customer_id'].unique()
    
    # Productos disponibles del catálogo real
    productos_disponibles = product_category_df['v.Code_pr'].unique()
    categorias_disponibles = categories_df['category_id'].unique()
    nombres_categorias = categories_df.set_index('category_id')['category_name'].to_dict()
    
    # Crear lista para almacenar registros
    registros_sinteticos = []
    
    # Generar customer_ids sintéticos que no existan en los datos reales
    customer_max = customer_ids_real.max() if len(customer_ids_real) > 0 else 1000000
    customer_ids_sinteticos = range(customer_max + 1, customer_max + 1 + n_registros + 1000)
    
    for i in range(n_registros):
        # Generar datos aleatorios dentro de los rangos reales
        fecha = fechas_min + timedelta(
            days=random.randint(0, (fechas_max - fechas_min).days)
        )
        
        store_id = random.choice(store_ids)
        customer_id = random.choice(customer_ids_sinteticos)
        
        # Seleccionar productos aleatorios del catálogo real
        num_productos = random.randint(1, 5)  # Entre 1 y 5 productos por transacción
        productos_seleccionados = random.sample(list(productos_disponibles), num_productos)
        
        # Para cada producto, crear un registro
        for producto_id in productos_seleccionados:
            # Buscar la categoría del producto
            categoria_info = product_category_df[product_category_df['v.Code_pr'] == producto_id]
            
            if not categoria_info.empty:
                category_id = categoria_info['v.code'].iloc[0]
                category_name = nombres_categorias.get(category_id, 'Unknown')
                v_Code_pr = producto_id
                v_code = category_id
            else:
                category_id = None
                category_name = None
                v_Code_pr = producto_id
                v_code = None
            
            registro = {
                'date': fecha,
                'store_id': store_id,
                'customer_id': customer_id,
                'products': producto_id,  # Producto individual (ya explotado)
                'num_products': 1,  # Cada registro explotado tiene 1 producto
                'product_id': producto_id,
                'v.Code_pr': v_Code_pr,
                'v.code': v_code,
                'category_id': category_id,
                'category_name': category_name
            }
            
            registros_sinteticos.append(registro)
    
    # Crear DataFrame
    df_sintetico = pd.DataFrame(registros_sinteticos)
    
    # Asegurar que los tipos de datos coincidan
    df_sintetico['date'] = pd.to_datetime(df_sintetico['date'])
    df_sintetico['store_id'] = df_sintetico['store_id'].astype(int)
    df_sintetico['customer_id'] = df_sintetico['customer_id'].astype(int)
    df_sintetico['products'] = df_sintetico['products'].astype(str)
    df_sintetico['num_products'] = df_sintetico['num_products'].astype(int)
    df_sintetico['product_id'] = df_sintetico['product_id'].astype(int)
    
    return df_sintetico
