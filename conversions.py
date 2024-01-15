from __future__ import annotations
from typing import Callable, Dict, Generator, Tuple, Sequence
import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit

Image = Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]

class CircuitBuilder:

    def __init__(self, w_qubits: int, h_qubits: int) -> None:
        self.__w_qubits = w_qubits
        self.__h_qubits = h_qubits
        self.__data = [np.uint8(0)] * (2 ** self.qubits())
        self.__circuit = QuantumCircuit(self.qubits())
    
    def qubits(self) -> int:
        return self.__w_qubits + self.__h_qubits
    
    def from_circuit(self, circuit: QuantumCircuit) -> CircuitBuilder:
        assert circuit.num_qubits == self.qubits()
        self.__circuit = circuit
        return self
    
    def gates(self, function: Callable[[QuantumCircuit], None]) -> CircuitBuilder:
        function(self.__circuit)
        return self
    
    def with_data(self, data: Dict[str, np.complex128]) -> CircuitBuilder:
        assert len(data) == 2 ** self.qubits()
        for key, value in data.items():
            self.__data[int(key, 2)] = value
        return self
    
    def build(self) -> QuantumCircuit:
        self.__circuit.initialize(self.__data, range(self.qubits()))
        return self.__circuit

def _map_channel_to_states(channel: NDArray[np.uint8], w_strings: Sequence[str], h_strings: Sequence[str]) -> Dict[str, np.complex128]:
    grid: Dict[str, Tuple[int, int]] = {}

    for x in range(len(w_strings)):
        for y in range(len(h_strings)):
            grid[w_strings[x] + h_strings[y]] = (x, y)

    data = {bit_string: np.sqrt(channel[pos]) for bit_string, pos in grid.items()}
    norm = np.sqrt(np.sum(np.array(data.values()) ** 2))

    for key, value in data.items():
        data[key] = value / norm
    
    return data

def hamming_manhattan_mapping(n_qubits: int) -> Sequence[str]:
    bit_strings = ["0", "1"]
    for _ in range(n_qubits):
        pad_0 = [elem + "0" for elem in bit_strings]
        pad_1 = [elem + "1" for elem in bit_strings[::-1]]
        bit_strings = pad_0 + pad_1
    return bit_strings

def circuits(image: Image, pixel_mapping: Callable[[int], Sequence[str]] = hamming_manhattan_mapping) -> Generator[CircuitBuilder, None, None]:
    for channel in image:
        width, height = channel.shape
        w_qubits, h_qubits = np.ceil(np.log2(width), dtype=int), np.ceil(np.log2(height), dtype=int)
        data = _map_channel_to_states(channel, pixel_mapping(w_qubits), pixel_mapping(h_qubits))        
        yield CircuitBuilder(w_qubits, h_qubits).with_data(data)