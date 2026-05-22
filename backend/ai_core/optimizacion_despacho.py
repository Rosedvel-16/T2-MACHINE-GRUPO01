import pandas as pd

def optimizar_despachos(pedidos, capacidad_maxima_vehiculo):
    """
    Optimiza la asignación de pedidos a un vehículo de despacho usando un enfoque Greedy.
    - Se optimiza: El valor total despachado (Prioridad + Tiempo de espera)[cite: 133, 138].
    - Restricción: La capacidad de peso máximo del vehículo (kg)[cite: 139].
    """
    
    df = pd.DataFrame(pedidos)
    
    df['score'] = (df['urgencia'] * 10) + df['tiempo_espera_minutos']
    
    df['ratio'] = df['score'] / df['peso_kg']
    df_ordenado = df.sort_values(by='ratio', ascending=False)
    
    peso_actual = 0
    despachos_aprobados = []
    
    for index, pedido in df_ordenado.iterrows():
        if peso_actual + pedido['peso_kg'] <= capacidad_maxima_vehiculo:
            despachos_aprobados.append(pedido['id_pedido'])
            peso_actual += pedido['peso_kg']
            
    print(f"📦 Capacidad del vehículo: {capacidad_maxima_vehiculo} kg")
    print(f"✅ Pedidos seleccionados para despacho inmediato: {despachos_aprobados}")
    print(f"⚖️ Peso total utilizado: {peso_actual} kg")
    
    print("\nJustificación: Se priorizó una combinación de urgencia y tiempo de espera penalizada por el peso, asegurando que los paquetes más críticos y ligeros salgan primero para maximizar la cantidad de entregas importantes sin exceder la restricción de carga.")

    return despachos_aprobados

if __name__ == "__main__":
    # Simulación de la cola de pedidos del almacén
    cola_de_pedidos = [
        {'id_pedido': 'P001', 'peso_kg': 50, 'urgencia': 1, 'tiempo_espera_minutos': 120},
        {'id_pedido': 'P002', 'peso_kg': 10, 'urgencia': 3, 'tiempo_espera_minutos': 15},  # Muy urgente
        {'id_pedido': 'P003', 'peso_kg': 30, 'urgencia': 2, 'tiempo_espera_minutos': 60},
        {'id_pedido': 'P004', 'peso_kg': 80, 'urgencia': 1, 'tiempo_espera_minutos': 200}, # Pesado pero lleva mucho tiempo
        {'id_pedido': 'P005', 'peso_kg': 15, 'urgencia': 3, 'tiempo_espera_minutos': 5}
    ]
    
    # Restricción: El montacargas/vehículo solo aguanta 100kg
    optimizar_despachos(cola_de_pedidos, capacidad_maxima_vehiculo=100)