import numpy as np
from qiskit import QuantumCircuit

def rx_gates(phi, qc: QuantumCircuit): # intermediate between the image and the same flipped
    qc.rx(phi, range(qc.num_qubits))

    # IMPORTANT: NO IMAGE WAS OUTPUT WHEN USING PI/16!! TODO: LOOK INTO WHY

def ry_gates(qc: QuantumCircuit): # intermediate between the image and the same flipped
    qc.ry(np.pi / 8, range(qc.num_qubits))

    # IMPORTANT: A BLACK IMAGE WAS OUTPUT WHEN USING PI/8!! TODO: LOOK INTO WHY

def single_rx_gate(qc: QuantumCircuit): # intermediate between the image and the same flipped
    qc.rx(np.pi/2, 4)

    # IMPORTANT: OUTPUT DIDN'T SEEM TO CHANGE FOR PI/16! TODO: LOOK INTO WHY
    # possible explanation: the effect is barely visible

def rz_gates(qc: QuantumCircuit): 
    qc.rz(np.pi / 4, range(qc.num_qubits))

def single_hadamard(qc: QuantumCircuit): # vertical bars of size 2^{qubit with hadamard gate}
    qc.h(4)
    # qc.rx(-np.pi/3, 0)

def some_mix_1(qc: QuantumCircuit): 
    qc.h(0)
    qc.rx(-np.pi/3, 4)

def hadamard_last(qc: QuantumCircuit):
    qc.h(qc.num_qubits-1)

def rx_last_qubit(qc: QuantumCircuit):
    qc.rx(np.pi, qc.num_qubits-1)

def swap_circuit_first_last(qc: QuantumCircuit):
    qc.swap(0, qc.num_qubits-2)

def rotate_by_angle(phi):
    '''
    Having this wrapper that returns the actual gate sequence allows us to change the phi angle dynamically.
    Change the inner method as needed.
    :param phi: the angle in radians
    '''
    
    return lambda x: rx_gates(phi, x)
