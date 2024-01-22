from typing import Sequence, Any

import cv2
import numpy as np
from apng import APNG
from qiskit import QuantumCircuit
from typing import Callable
from qiskit import QuantumCircuit, QuantumRegister

from circuit_conversion import channel_to_circuit, image_to_circuits, probabilities_to_channel, run_circuit
from image_preprocessing import image_read


def rx_gates(phi, qc: QuantumCircuit): # intermediate between the image and the same flipped
    qc.rx(phi, range(qc.num_qubits))

    # IMPORTANT: NO IMAGE WAS OUTPUT WHEN USING PI/16!! TODO: LOOK INTO WHY

def ry_gates(phi: float, qc: QuantumCircuit): # intermediate between the image and the same flipped
    qc.ry(phi, range(qc.num_qubits))

    # IMPORTANT: A BLACK IMAGE WAS OUTPUT WHEN USING PI/8!! TODO: LOOK INTO WHY

def single_rx_gate(phi: float, i: int, qc: QuantumCircuit): # intermediate between the image and the same flipped
    qc.rx(phi, i)

    # IMPORTANT: OUTPUT DIDN'T SEEM TO CHANGE FOR PI/16! TODO: LOOK INTO WHY
    # possible explanation: the effect is barely visible

def rz_gates(phi: float, qc: QuantumCircuit):
    qc.rz(phi, range(qc.num_qubits))

def single_hadamard(i: int, qc: QuantumCircuit): # vertical bars of size 2^{qubit with hadamard gate}
    qc.h(i)

def some_mix_1(qc: QuantumCircuit):
    qc.h(0)
    qc.rx(-np.pi/3, 4)

def swap_gate(i: int, j: int, qc: QuantumCircuit):
    qc.swap(i, j)

def rotate_by_angle(phi):
    '''
    Having this wrapper that returns the actual gate sequence allows us to change the phi angle dynamically.
    Change the inner method as needed.
    :param phi: the angle in radians
    '''
    def inner_method(qc: QuantumCircuit, registers: Sequence[QuantumRegister]):
        qc.rx(phi, qc.qubits)
    return inner_method


def animate_image(
        filename: str,
        output_filename: str,
        frames: int, fps=1000//24,
        grayscale=False
):
    if grayscale:
        circuit_builders = [channel_to_circuit(image_read(filename, grayscale=True))]
    else:
        circuit_builders = [circ for circ in image_to_circuits(image_read(filename))]

    files = []
    for i in range(frames):
        circuits = [cb.gates(rotate_by_angle(2 * np.pi * i / fps)).build() for cb in circuit_builders]
        channels = [probabilities_to_channel(run_circuit(qc)) for qc in circuits]

        files.append(f'media/{i}.png')
        cv2.imwrite(f'media/{i}.png', np.stack(channels, axis=2))
    APNG.from_files(files, delay=fps).save(output_filename)


def apply_gate_to_image(
        filename: str,
        output_filename: str,
        gate: Callable[[QuantumCircuit], Any],
        grayscale=False
):
    if grayscale:
        circuit_builders = [channel_to_circuit(image_read(filename, grayscale=True))]
    else:
        circuit_builders = [circ for circ in image_to_circuits(image_read(filename))]

    circuits = [cb.gates(gate).build() for cb in circuit_builders]
    channels = [probabilities_to_channel(run_circuit(qc)) for qc in circuits]

    cv2.imwrite(output_filename, np.stack(channels, axis=2))
