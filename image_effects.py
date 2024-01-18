import numpy as np
from qiskit import QuantumCircuit
from typing import Callable

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
    
    return lambda circuit, registers: rx_gates(phi, circuit)
