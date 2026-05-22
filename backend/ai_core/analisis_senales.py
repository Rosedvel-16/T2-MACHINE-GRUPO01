import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def analizar_senales(archivo_csv='data/simulacion_almacen.csv'):
    """
    Analiza señales temporales de vibración y temperatura [cite: 31-34].
    """
    df = pd.read_csv(archivo_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # --- Análisis de Señal 1: Vibración --- [cite: 33]
    # Detectar picos que superen los 20Hz (umbral de peligro)
    picos_vib, _ = find_peaks(df['vibracion'], height=20)

    # --- Análisis de Señal 2: Temperatura --- [cite: 34]
    # Detectar picos que superen los 30°C
    picos_temp, _ = find_peaks(df['temperatura'], height=30)

    # Mostrar conclusiones (Esto va para el informe) [cite: 37]
    print(f"🚨 Anomalías detectadas en Vibración: {len(picos_vib)} picos. Implicación: Posible daño mecánico en rodamientos de la faja transportadora.")
    print(f"🔥 Anomalías detectadas en Temperatura: {len(picos_temp)} picos. Implicación: Riesgo de sobrecalentamiento en zona de servidores o falla del sistema HVAC.")

    # Graficar las señales [cite: 36]
    plt.figure(figsize=(12, 6))

    # Gráfico de Vibración
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['vibracion'], label='Vibración (Hz)', color='blue')
    plt.plot(df['timestamp'].iloc[picos_vib], df['vibracion'].iloc[picos_vib], "x", color='red', markersize=10, label='Anomalías (Picos)')
    plt.axhline(y=20, color='r', linestyle='--', label='Umbral Peligro')
    plt.title('Análisis Temporal de Vibración')
    plt.legend()

    # Gráfico de Temperatura
    plt.subplot(2, 1, 2)
    plt.plot(df['timestamp'], df['temperatura'], label='Temperatura (°C)', color='orange')
    plt.plot(df['timestamp'].iloc[picos_temp], df['temperatura'].iloc[picos_temp], "x", color='red', markersize=10, label='Picos de Calor')
    plt.title('Análisis Temporal de Temperatura')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analizar_senales()