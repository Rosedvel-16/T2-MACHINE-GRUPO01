"""
PARTE J — CONCURRENCIA
Implementa ejecución concurrente de agentes usando threading y colas.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agentes import (
	Mensaje,
	AgenteSensor,
	AgenteAnalizadorSenales,
	AgenteAnalizadorImagenes,
	AgenteDecisor,
	AgenteCoordinador,
)


@dataclass
class PipelineResultado:
	mensajes: List[Mensaje] = field(default_factory=list)
	plan_global: Dict[str, object] = field(default_factory=dict)
	prioridad_global: int = 1
	ciclos: int = 0


class MessageBus:
	def __init__(self):
		self.queues: Dict[str, queue.Queue] = {}

	def register(self, name: str) -> None:
		if name not in self.queues:
			self.queues[name] = queue.Queue()

	def publish(self, mensaje: Mensaje) -> None:
		destino = mensaje.destino
		if destino == "BROADCAST":
			for name, q in self.queues.items():
				if name != mensaje.origen:
					q.put(mensaje)
			return
		if destino in self.queues:
			self.queues[destino].put(mensaje)

	def get(self, name: str, timeout: float = 0.2) -> Optional[Mensaje]:
		try:
			return self.queues[name].get(timeout=timeout)
		except queue.Empty:
			return None


class SensorWorker(threading.Thread):
	def __init__(self, bus: MessageBus, stop_event: threading.Event,
				 ciclos: int, intervalo: float = 0.3,
				 csv_sensores: Optional[str] = None):
		super().__init__(daemon=True)
		self.bus = bus
		self.stop_event = stop_event
		self.intervalo = intervalo
		self.ciclos = ciclos
		self.sensor = AgenteSensor(csv_sensores)
		self.historial: List[Mensaje] = []

	def run(self) -> None:
		for _ in range(self.ciclos):
			if self.stop_event.is_set():
				break
			mensaje = self.sensor.procesar(None)
			self.historial.append(mensaje)
			self.bus.publish(mensaje)
			time.sleep(self.intervalo)


class AgentWorker(threading.Thread):
	def __init__(self, name: str, agent, bus: MessageBus,
				 stop_event: threading.Event, out_route: Optional[str] = None,
				 expect_tipo: Optional[str] = None):
		super().__init__(daemon=True)
		self.name = name
		self.agent = agent
		self.bus = bus
		self.stop_event = stop_event
		self.out_route = out_route
		self.expect_tipo = expect_tipo
		self.historial: List[Mensaje] = []

	def run(self) -> None:
		while not self.stop_event.is_set():
			mensaje = self.bus.get(self.name, timeout=0.2)
			if mensaje is None:
				continue
			if self.expect_tipo and mensaje.tipo != self.expect_tipo:
				continue
			salida = self.agent.procesar(mensaje)
			if salida is None:
				continue
			self.historial.append(salida)
			if self.out_route:
				salida.destino = self.out_route
			self.bus.publish(salida)


class VisionScheduler(threading.Thread):
	def __init__(self, bus: MessageBus, stop_event: threading.Event,
				 ciclos: int, intervalo: float = 0.6):
		super().__init__(daemon=True)
		self.bus = bus
		self.stop_event = stop_event
		self.ciclos = ciclos
		self.intervalo = intervalo

	def run(self) -> None:
		for _ in range(self.ciclos):
			if self.stop_event.is_set():
				break
			msg = Mensaje(
				origen="Scheduler",
				destino="AgenteAnalizadorImagenes",
				tipo="comando",
				contenido={"accion": "analizar_imagenes"},
				prioridad=2,
			)
			self.bus.publish(msg)
			time.sleep(self.intervalo)


class FusionWorker(threading.Thread):
	def __init__(self, bus: MessageBus, stop_event: threading.Event,
				 incluir_vision: bool = True):
		super().__init__(daemon=True)
		self.bus = bus
		self.stop_event = stop_event
		self.incluir_vision = incluir_vision
		self.buffer_senales: Optional[Mensaje] = None
		self.buffer_vision: Optional[Mensaje] = None

	def run(self) -> None:
		while not self.stop_event.is_set():
			mensaje = self.bus.get("Fusionador", timeout=0.2)
			if mensaje is None:
				continue
			if mensaje.origen == "AgenteAnalizadorSenales":
				self.buffer_senales = mensaje
			elif mensaje.origen == "AgenteAnalizadorImagenes":
				self.buffer_vision = mensaje

			if self.buffer_senales and (self.buffer_vision or not self.incluir_vision):
				hechos: List[tuple] = []
				hechos.extend(self.buffer_senales.contenido.get("hechos", []))
				if self.buffer_vision:
					hechos.extend(self.buffer_vision.contenido.get("hechos", []))

				fusion = Mensaje(
					origen="FUSION",
					destino="AgenteDecisor",
					tipo="analisis",
					contenido={"hechos": hechos},
					prioridad=2,
				)
				self.bus.publish(fusion)
				self.buffer_senales = None
				self.buffer_vision = None


class CoordinadorWorker(threading.Thread):
	def __init__(self, bus: MessageBus, stop_event: threading.Event):
		super().__init__(daemon=True)
		self.bus = bus
		self.stop_event = stop_event
		self.coordinador = AgenteCoordinador()
		self.ultimo_plan: Dict[str, object] = {}
		self.ultima_prioridad: int = 1
		self.historial: List[Mensaje] = []

	def run(self) -> None:
		while not self.stop_event.is_set():
			mensaje = self.bus.get("AgenteCoordinador", timeout=0.2)
			if mensaje is None:
				continue
			salida = self.coordinador.procesar(mensaje)
			if salida is None:
				continue
			self.historial.append(salida)
			self.ultimo_plan = salida.contenido
			self.ultima_prioridad = salida.prioridad


def ejecutar_concurrente(ciclos: int = 3, incluir_vision: bool = True,
						 intervalo_sensor: float = 0.3,
						 intervalo_vision: float = 0.6,
						 csv_sensores: Optional[str] = None) -> PipelineResultado:
	"""Ejecuta el pipeline multiagente con concurrencia real usando threading."""
	bus = MessageBus()
	for name in [
		"AgenteAnalizadorSenales",
		"AgenteAnalizadorImagenes",
		"Fusionador",
		"AgenteDecisor",
		"AgenteCoordinador",
	]:
		bus.register(name)

	stop_event = threading.Event()

	sensor_worker = SensorWorker(
		bus, stop_event, ciclos=ciclos, intervalo=intervalo_sensor, csv_sensores=csv_sensores
	)
	scheduler = VisionScheduler(bus, stop_event, ciclos=ciclos, intervalo=intervalo_vision)

	an_senales_worker = AgentWorker(
		"AgenteAnalizadorSenales",
		AgenteAnalizadorSenales(),
		bus,
		stop_event,
		out_route="Fusionador",
		expect_tipo="lectura",
	)
	an_imagenes_worker = AgentWorker(
		"AgenteAnalizadorImagenes",
		AgenteAnalizadorImagenes(),
		bus,
		stop_event,
		out_route="Fusionador",
		expect_tipo="comando",
	)
	fusion_worker = FusionWorker(bus, stop_event, incluir_vision=incluir_vision)

	decisor_worker = AgentWorker(
		"AgenteDecisor",
		AgenteDecisor(),
		bus,
		stop_event,
		out_route="AgenteCoordinador",
		expect_tipo="analisis",
	)
	coordinador_worker = CoordinadorWorker(bus, stop_event)

	threads = [
		sensor_worker,
		scheduler,
		an_senales_worker,
		an_imagenes_worker,
		fusion_worker,
		decisor_worker,
		coordinador_worker,
	]

	for t in threads:
		t.start()

	sensor_worker.join()
	scheduler.join()

	# Esperar a que los últimos mensajes lleguen al coordinador
	time.sleep(max(intervalo_sensor, intervalo_vision))
	stop_event.set()

	resultado = PipelineResultado(
		mensajes=(
			sensor_worker.historial
			+ an_senales_worker.historial
			+ an_imagenes_worker.historial
			+ coordinador_worker.historial
		),
		plan_global=coordinador_worker.ultimo_plan,
		prioridad_global=coordinador_worker.ultima_prioridad,
		ciclos=ciclos,
	)
	return resultado


if __name__ == "__main__":
	salida = ejecutar_concurrente(ciclos=3, incluir_vision=True)
	print("Plan global:", salida.plan_global)