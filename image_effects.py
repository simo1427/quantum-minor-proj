from ast import Call
from typing import Any, Optional

import cv2
import numpy as np
from apng import APNG
from qiskit import QuantumCircuit
from qiskit.extensions import UnitaryGate
from typing import Callable
from animation_curves import linear

from circuit_conversion import Effect, channel_to_circuit, image_to_circuits, probabilities_to_channel, run_circuit
from image_preprocessing import image_read


# IMPORTANT: all effect functions must have a signature of the Effect class


# intermediate between the image and the same flipped
def rx_gates(circuit: QuantumCircuit, phi: float, **kwargs) -> None:
    circuit.rx(phi, range(circuit.num_qubits))

    # IMPORTANT: NO IMAGE WAS OUTPUT WHEN USING PI/16!! TODO: LOOK INTO WHY


# intermediate between the image and the same flipped
def ry_gates(circuit: QuantumCircuit, phi: float, **kwargs) -> None:
    circuit.ry(phi, range(circuit.num_qubits))

    # IMPORTANT: A BLACK IMAGE WAS OUTPUT WHEN USING PI/8!! TODO: LOOK INTO WHY


# intermediate between the image and the same flipped
def single_rx_gate(circuit: QuantumCircuit, i: int, phi: float, **kwargs) -> None:
    circuit.rx(phi, i)

    # IMPORTANT: OUTPUT DIDN'T SEEM TO CHANGE FOR PI/16! TODO: LOOK INTO WHY
    # possible explanation: the effect is barely visible


def rz_gates(circuit: QuantumCircuit, phi: float, **kwargs) -> None:
    circuit.rz(phi, range(circuit.num_qubits))


# vertical bars of size 2^{qubit with hadamard gate}
def single_hadamard(circuit: QuantumCircuit, i: int, **kwargs) -> None:
    circuit.h(i)


def some_mix_1(circuit: QuantumCircuit, **kwargs) -> None:
    circuit.h(0)
    circuit.rx(-np.pi/3, 4)


def swap_gate(circuit: QuantumCircuit, i: int, j: int, **kwargs) -> None:
    circuit.swap(i, j)


# partially swaps amplitudes two registers (images) based on https://en.wikipedia.org/wiki/List_of_quantum_logic_gates#Non-Clifford_swap_gates
def partial_swap(circuit: QuantumCircuit, alpha: float, **kwargs) -> None:
    gate = UnitaryGate(
        np.array([[1,                                    0,                                    0, 0],
                  [0, (1 + np.exp(1j * np.pi * alpha)) / 2, (1 - np.exp(1j * np.pi * alpha)) / 2, 0],
                  [0, (1 - np.exp(1j * np.pi * alpha)) / 2, (1 + np.exp(1j * np.pi * alpha)) / 2, 0],
                  [0,                                    0,                                    0, 1]]),
        label="SWAP^alpha gate"
    )
    circuit.append(gate, [circuit.qregs[0], circuit.qregs[1]])


# def partial_swap(circuit: QuantumCircuit, percentage: float, **kwargs) -> None:
#     circuit.rxx(np.pi / 2 * percentage, circuit.qregs[0], circuit.qregs[1])
#     circuit.ryy(np.pi / 2 * percentage, circuit.qregs[0], circuit.qregs[1])
#     circuit.rzz(np.pi / 2 * percentage, circuit.qregs[0], circuit.qregs[1])


def animate_image(
        filename: str,
        output_filename: str,
        frames: int, 
        fps: int = 24,
        animation_curve: Callable[[float], float] = linear,
        grayscale: bool = False,
        shots: Optional[int] = None
):
    if grayscale:
        circuit_builders = [channel_to_circuit(image_read(filename, grayscale=True))]
    else:
        circuit_builders = [circ for circ in image_to_circuits(image_read(filename))]

    files = []
    for i in range(frames):
        t = animation_curve(i / (frames - 1))
        circuits = [cb.apply_effect(rx_gates, phi=np.pi * t).build() for cb in circuit_builders]
        channels = np.array([list(probabilities_to_channel(run_circuit(qc, shots))) for qc in circuits]).squeeze(axis=1)
        files.append(f'media/{i}.png')
        cv2.imwrite(f'media/{i}.png', cv2.resize(np.stack(channels, axis=2), (256, 256), interpolation=cv2.INTER_NEAREST))
    APNG.from_files(files, delay=1000//fps).save(output_filename)


def apply_effect_to_image(
        filename: str,
        output_filename: str,
        effect: Effect,
        grayscale=False,
        **kwargs
):
    if grayscale:
        circuit_builders = [channel_to_circuit(image_read(filename, grayscale=True))]
    else:
        circuit_builders = [circ for circ in image_to_circuits(image_read(filename))]

    circuits = [cb.apply_effect(effect, **kwargs).build() for cb in circuit_builders]
    channels = np.array([list(probabilities_to_channel(run_circuit(qc))) for qc in circuits]).squeeze(axis=1)

    cv2.imwrite(output_filename, cv2.resize(np.stack(channels, axis=2), (256, 256), interpolation=cv2.INTER_NEAREST))
