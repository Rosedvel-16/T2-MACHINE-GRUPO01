# Informe técnico — T2 Machine Learning

> Cada integrante redacta su parte. La sección **8 — Sistema experto**, **9 — Representación
> del conocimiento** y **10 — Diseño de agentes** corresponde a la **Persona 3
> (Arquitecto de la inteligencia del sistema)**.

---

## 8. Sistema experto y reglas (Parte D)

### 8.1 Descripción

El módulo `backend/ai_core/sistema_experto.py` implementa un sistema experto
clásico basado en **reglas de producción (if … then)** con **motor de inferencia
hacia adelante** (forward chaining). El sistema modela el conocimiento
operativo del almacén: a partir de hechos observados por los sensores
(Parte A — Gaby) y por el módulo de visión (Parte C — Cano), deriva
conclusiones y emite acciones recomendadas.

Se cumple con holgura el requisito mínimo de la rúbrica:

| Requisito                | Mínimo exigido | Implementado |
|--------------------------|:--------------:|:------------:|
| Hechos                   | 8              | **10**       |
| Reglas if/then           | 6              | **8**        |
| Consultas y conclusiones | Sí             | **Sí (con explicación)** |

### 8.2 Hechos definidos

Los 10 hechos atómicos cubren los cuatro dominios del problema:

| Hecho                     | Dominio    | Descripción                                          |
|---------------------------|------------|------------------------------------------------------|
| `temperatura_alta`        | ambiental  | Temperatura interna > 30 °C                          |
| `humedad_excesiva`        | ambiental  | Humedad relativa > 80 %                              |
| `vibracion_anomala`       | sensores   | Vibración en fajas > umbral seguro                   |
| `zona_saturada`           | logística  | Ocupación de zona de carga > 90 %                    |
| `paquete_danado`          | visión     | El módulo de visión detecta daño en el paquete       |
| `ruta_obstruida`          | visión     | El pasillo principal está bloqueado                  |
| `prioridad_alta_pedido`   | logística  | Existe un pedido VIP en cola                         |
| `sensor_fallido`          | sensores   | Un sensor lleva varios ciclos sin reportar           |
| `zona_libre`              | logística  | Zona de despacho < 40 % de ocupación                 |
| `mantenimiento_diferido`  | operativo  | No se realizó mantenimiento preventivo a tiempo      |

### 8.3 Reglas if/then

| ID    | Regla                                                                                         | Prioridad |
|-------|-----------------------------------------------------------------------------------------------|:---------:|
| R-01  | **IF** `vibracion_anomala` **THEN** `riesgo_mecanico`                                         | 3         |
| R-02  | **IF** `temperatura_alta` ∧ `humedad_excesiva` **THEN** `alerta_ambiental`                    | 3         |
| R-03  | **IF** `paquete_danado` **THEN** `revision_manual_paquete`                                    | 2         |
| R-04  | **IF** `zona_saturada` ∧ `ruta_obstruida` **THEN** `reubicar_carga`                           | 3         |
| R-05  | **IF** `prioridad_alta_pedido` ∧ `zona_libre` **THEN** `despacho_inmediato`                   | 2         |
| R-06  | **IF** `riesgo_mecanico` ∧ `mantenimiento_diferido` **THEN** `parada_emergencia`              | 4         |
| R-07  | **IF** `sensor_fallido` **THEN** `datos_no_confiables`                                        | 2         |
| R-08  | **IF** `alerta_ambiental` ∧ `zona_saturada` **THEN** `riesgo_mercancia`                       | 3         |

Las reglas **R-06** y **R-08** son **reglas derivadas en cascada**: usan como
antecedente conclusiones de otras reglas (`riesgo_mecanico` viene de R-01;
`alerta_ambiental` viene de R-02). Esto demuestra que el motor encadena
correctamente la inferencia hasta el punto fijo.

### 8.4 Motor de inferencia

El algoritmo es **forward chaining clásico**:

1. Cargar los hechos iniciales en la base.
2. Recorrer todas las reglas; para cada una, comprobar si todos sus
   antecedentes están activos.
3. Cuando una regla se dispara, asertar su consecuente como nuevo hecho.
4. Repetir hasta que ningún ciclo nuevo dispare reglas adicionales (punto fijo).
5. Ordenar las conclusiones por prioridad y emitir.

Adicionalmente se calcula una **confianza combinada** por la **regla del mínimo**
(estilo MYCIN): `conf(consecuente) = min(conf(antecedentes))`. Esto permite
propagar la incertidumbre proveniente de la Parte H (razonamiento
probabilístico).

### 8.5 Consultas y explicación

Cada conclusión es **trazable**. El método `motor.explicar(hecho)` reconstruye
la cadena de reglas que llevaron al sistema a derivar ese hecho, por ejemplo:

```
Explicación de 'parada_emergencia':
  • Regla R-06: riesgo_mecanico ∧ mantenimiento_diferido → parada_emergencia (95 %)
  • Regla R-01: vibracion_anomala → riesgo_mecanico (95 %)
  • vibracion_anomala: hecho directo de 'escenario' (95 %)
```

Esta auditoría es clave en sistemas de supervisión industrial porque el
operador necesita justificar cada alarma.

---

## 9. Representación del conocimiento (Parte E)

### 9.1 Estructuras de datos utilizadas

| Elemento     | Estructura Python              | Justificación                                        |
|--------------|--------------------------------|------------------------------------------------------|
| Hecho        | `@dataclass Hecho`             | Campos tipados + metadatos (origen, confianza, ts)   |
| Regla        | `@dataclass Regla` + `lambda`  | Antecedentes declarativos + condición extra opcional |
| Base hechos  | `Dict[str, Hecho]`             | Acceso O(1) por nombre                                |
| Base reglas  | `List[Regla]`                  | Iteración ordenada por prioridad                      |
| Conclusión   | `@dataclass Conclusion`        | Registro inmutable con trazabilidad temporal          |

El uso de `@dataclass` (en lugar de diccionarios planos) garantiza
**autocompletado** en el IDE, **validación estática** y **serialización
homogénea** a JSON para el dashboard.

### 9.2 Tabla de conocimiento (hechos × reglas)

La función `construir_tabla_conocimiento()` produce una matriz que documenta
**en qué reglas interviene cada hecho** y si es **primitivo** (introducido por
un sensor/agente) o **derivado** (concluido por el motor). Esta es la
evidencia visual exigida por la Parte E.

| Hecho                  | Dominio    | Tipo       | Reglas que lo usan    |
|------------------------|------------|------------|-----------------------|
| `temperatura_alta`     | ambiental  | primitivo  | R-02                  |
| `humedad_excesiva`     | ambiental  | primitivo  | R-02                  |
| `vibracion_anomala`    | sensores   | primitivo  | R-01                  |
| `paquete_danado`       | visión     | primitivo  | R-03                  |
| `zona_saturada`        | logística  | primitivo  | R-04, R-08            |
| `ruta_obstruida`       | visión     | primitivo  | R-04                  |
| `prioridad_alta_pedido`| logística  | primitivo  | R-05                  |
| `zona_libre`           | logística  | primitivo  | R-05                  |
| `sensor_fallido`       | sensores   | primitivo  | R-07                  |
| `mantenimiento_diferido`| operativo | primitivo  | R-06                  |
| `riesgo_mecanico`      | derivado   | derivado   | R-06                  |
| `alerta_ambiental`     | derivado   | derivado   | R-08                  |
| `revision_manual_paquete`| derivado | derivado   | —                     |
| `reubicar_carga`       | derivado   | derivado   | —                     |
| `despacho_inmediato`   | derivado   | derivado   | —                     |
| `parada_emergencia`    | derivado   | derivado   | —                     |
| `datos_no_confiables`  | derivado   | derivado   | —                     |
| `riesgo_mercancia`     | derivado   | derivado   | —                     |

### 9.3 Relaciones causales

El diccionario `RELACIONES_CAUSALES` documenta las cadenas típicas de
razonamiento. Sirve tanto para fines didácticos como para que el dashboard
pinte el **grafo de inferencia**:

```python
RELACIONES_CAUSALES = {
    "vibracion_anomala"      : ["riesgo_mecanico", "parada_emergencia"],
    "temperatura_alta"       : ["alerta_ambiental", "riesgo_mercancia"],
    "humedad_excesiva"       : ["alerta_ambiental", "riesgo_mercancia"],
    "paquete_danado"         : ["revision_manual_paquete"],
    "zona_saturada"          : ["reubicar_carga", "riesgo_mercancia"],
    "ruta_obstruida"         : ["reubicar_carga"],
    "prioridad_alta_pedido"  : ["despacho_inmediato"],
    "sensor_fallido"         : ["datos_no_confiables"],
    "mantenimiento_diferido" : ["parada_emergencia"],
}
```

### 9.4 Esquema gráfico

```
                ┌───────────────────────┐
                │  Hechos PRIMITIVOS    │
                │ (sensores + visión)   │
                └──────────┬────────────┘
                           │
                           ▼  forward chaining
                ┌───────────────────────┐
                │  Motor de Inferencia  │   ← Reglas R-01..R-08
                │   (R-01 … R-08)       │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │  Hechos DERIVADOS     │
                │ (conclusiones +       │
                │  acciones priorizadas)│
                └───────────────────────┘
```

---

## 10. Diseño de agentes (Parte F)

### 10.1 Arquitectura general

El módulo `backend/ai_core/agentes.py` implementa **5 agentes** (uno más del
mínimo de 4 requerido por la rúbrica) que se comunican mediante objetos
`Mensaje` con un protocolo común. Esto cumple simultáneamente:

- Parte F (diseño de agentes).
- Parte I (comunicación entre agentes — apoyo a Beckam).
- Habilita la Parte J (concurrencia con `threading`) porque cada agente
  expone `procesar(mensaje)` independiente del transporte.

### 10.2 Especificación de cada agente

| Agente                     | Conocimiento                                       | Entrada                            | Salida                                  | Cómo decide                                          |
|----------------------------|----------------------------------------------------|-------------------------------------|-----------------------------------------|------------------------------------------------------|
| **AgenteSensor**           | umbrales nominales + sensores disponibles          | (productor)                         | `Mensaje("lectura")` con variables      | muestrea del CSV (Parte A) o genera lectura simulada |
| **AgenteAnalizadorSeñales**| umbrales + ventana móvil de 60 lecturas            | `Mensaje("lectura")`                | `Mensaje("analisis")` con hechos        | umbralización numérica → hechos lógicos              |
| **AgenteAnalizadorImagenes**| reporte JSON de `vision_cv` (Parte C)             | comando o evento periódico          | `Mensaje("analisis")` con hechos visión | traduce niveles de alerta visual a hechos            |
| **AgenteDecisor**          | 10 hechos + 8 reglas del sistema experto           | `Mensaje("analisis")`               | `Mensaje("decision")` priorizadas       | forward chaining sobre las reglas if/then            |
| **AgenteCoordinador**      | política global + motor probabilístico (Parte H)   | `Mensaje("decision")`               | `Mensaje("alerta")` con plan global     | fusión Noisy-OR + clasificación por nivel            |

### 10.3 Protocolo de comunicación

Todos los agentes intercambian instancias de la dataclass `Mensaje`:

```python
@dataclass
class Mensaje:
    origen      : str            # Agente emisor
    destino     : str            # Agente destinatario ("BROADCAST" si va a todos)
    tipo        : str            # lectura | analisis | alerta | decision | comando
    contenido   : Dict[str, Any] # Payload (datos, hechos, conclusiones)
    prioridad   : int            # 1=baja, 2=media, 3=alta, 4=crítica
    timestamp   : str
    id_mensaje  : str
```

Este protocolo es **transporte-agnóstico**: puede viajar por una
`queue.Queue` (Parte J — `threading`), por una pipe de `multiprocessing` o
por un endpoint HTTP, sin modificar la lógica interna de los agentes.

### 10.4 Diagrama de interacción

```
   ┌──────────────┐ lectura  ┌────────────────────────┐
   │ AgenteSensor │──────────▶│ AgenteAnalizadorSenales│
   └──────────────┘           └─────────────┬──────────┘
                                            │ analisis (hechos)
   ┌─────────────────────────┐ analisis     │
   │ AgenteAnalizadorImagenes│──────────────┤
   └─────────────────────────┘              ▼
                                  ┌───────────────────┐
                                  │  AgenteDecisor    │   (motor experto)
                                  └─────────────┬─────┘
                                                │ decision
                                                ▼
                                  ┌───────────────────┐
                                  │ AgenteCoordinador │   (Noisy-OR + plan)
                                  └─────────────┬─────┘
                                                │ alerta BROADCAST
                                                ▼
                                          (Dashboard Streamlit)
```

### 10.5 Integración con el resto del proyecto

| Necesita                          | De qué módulo                       | Cómo lo consume                                |
|-----------------------------------|-------------------------------------|------------------------------------------------|
| Datos de sensores (Parte A)       | `data/simulacion_almacen.csv`       | `AgenteSensor` lee el CSV con `pandas`         |
| Reporte de visión (Parte C)       | `images_output/reporte_vision.json` | `AgenteAnalizadorImagenes` lee el JSON         |
| Fusión probabilística (Parte H)   | `logica_probabilidad.py`            | `AgenteCoordinador` usa `MotorProbabilistico`  |
| Reglas if/then (Parte D)          | `sistema_experto.py`                | `AgenteDecisor` ejecuta `MotorExperto.inferir`|
| Concurrencia (Parte J)            | `concurrencia.py` (Beckam)          | Cada agente expone `procesar(mensaje)`         |

### 10.6 Ejemplo de ejecución

Ejecutando `python agentes.py` se obtiene la siguiente traza (ciclo #1 con
datos reales del CSV y con un reporte simulado de visión):

```
Plan global emitido por el Coordinador:
  Nivel de riesgo : CRÍTICO  (88.0%)
  Directiva       : DETENER OPERACIONES — protocolo de emergencia.
  Acciones priorizadas:
    [P2] Retirar paquete de la línea y derivar a estación de inspección.
         (R-03 → revision_manual_paquete)
```

El reporte JSON completo se guarda en
`backend/ai_core/images_output/reporte_agentes.json` y queda disponible para
el dashboard y para anexar a las diapositivas.

---

## Notas de ejecución (sólo para esta sección)

### Probar las Partes D, E, F desde consola

```powershell
cd backend\ai_core
$env:PYTHONIOENCODING="utf-8"     # solo en Windows
python sistema_experto.py         # Parte D + E
python agentes.py                  # Parte F
```

### Probar las Partes D, E, F en el dashboard

```powershell
cd backend\ai_core
streamlit run streamlit_app.py
```

Y abrir las pestañas **Parte D – Sistema Experto**, **Parte E – Conocimiento**
y **Parte F – Agentes** en `http://localhost:8501`.
