import pandas as pd
import numpy as np

from paths import CSV_SENSORES, asegurar_carpetas


def generar_datos_almacen(num_registros=150, archivo_salida=None):
    if archivo_salida is None:
        asegurar_carpetas()
        archivo_salida = str(CSV_SENSORES)
    """
    Genera datos simulados del almacén logístico.
    Cumple con los requisitos de la Parte A [cite: 21-28].
    """
    # Tiempo simulado (ej. cada minuto)
    tiempo = pd.date_range(start='2026-05-21 08:00', periods=num_registros, freq='min')
    
    # 1. Temperatura (°C): Base 20°C con variación normal [cite: 23]
    temperatura = np.random.normal(20, 2, num_registros)
    # Inyectar anomalía: Un pico de calor repentino
    temperatura[50:55] = [35, 38, 40, 39, 36] 
    
    # 2. Humedad (%): Base 50% con variación normal [cite: 24]
    humedad = np.random.normal(50, 5, num_registros)
    
    # 3. Vibración (Hz): Fajas transportadoras base 10Hz [cite: 25]
    vibracion = np.random.normal(10, 1.5, num_registros)
    # Inyectar anomalía: Picos de vibración mecánica
    vibracion[80:83] = [25, 28, 24]
    vibracion[120] = 30
    
    # 4. Ocupación de zonas (%): Zonas de carga base 60% [cite: 25]
    ocupacion = np.random.randint(40, 85, num_registros)
    # Inyectar saturación
    ocupacion[100:110] = np.random.randint(95, 100, 10)

    # Crear DataFrame [cite: 28]
    df_almacen = pd.DataFrame({
        'timestamp': tiempo,
        'temperatura': temperatura,
        'humedad': humedad,
        'vibracion': vibracion,
        'ocupacion': ocupacion
    })

    # Guardar en CSV para que el resto de agentes lo consuma
    df_almacen.to_csv(archivo_salida, index=False)
    print(f"✅ Archivo '{archivo_salida}' generado con {num_registros} registros.")
    
    return df_almacen

if __name__ == "__main__":
    generar_datos_almacen()