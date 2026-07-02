import pandas as pd
import numpy as np

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Itera sobre las columnas de un dataframe y modifica el tipo de dato 
    para reducir el uso de memoria de forma segura.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Uso de memoria inicial del DataFrame: {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype
        
        # 1. Chequeo seguro para columnas exclusivamente numéricas
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Optimización de enteros
            if pd.api.types.is_integer_dtype(col_type):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            
            # Optimización de flotantes
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
        # 2. Optimización de variables categóricas (texto nativo u objetos)
        elif pd.api.types.is_string_dtype(col_type) or col_type == object:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Uso de memoria final tras optimización: {end_mem:.2f} MB')
        print(f'Reducción de memoria: {100 * (start_mem - end_mem) / start_mem:.1f}%')
        
    return df

def load_and_optimize_data(filepath: str) -> pd.DataFrame:
    """
    Carga el dataset y aplica la reducción de memoria inmediatamente.
    """
    print(f"Cargando datos desde: {filepath}...")
    df = pd.read_csv(filepath)
    df = reduce_mem_usage(df)
    return df