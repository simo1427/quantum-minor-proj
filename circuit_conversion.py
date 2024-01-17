from __future__ import annotations
from typing import Any, Callable, Mapping, Generator, Tuple, Sequence, MutableSequence
import numpy as np
from numpy.typing import NDArray
from qiskit import execute, Aer
from qiskit.circuit import QuantumCircuit
from qiskit.result import Result

Image = Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]
PixelMapping = Callable[[int], Sequence[str]]


class CircuitBuilder:
    '''
    Provides a nice interface for building quantum circuits tailored specifically for this project.
    '''
    def __init__(self, w_qubits: int, h_qubits: int) -> None:
        self.__w_qubits: int = w_qubits
        self.__h_qubits: int = h_qubits
        self.__data: MutableSequence[complex] = [0] * (2 ** self.qubits())
        self.__circuit: QuantumCircuit = QuantumCircuit(self.qubits())

    def qubits(self) -> int:
        return self.__w_qubits + self.__h_qubits

    def from_circuit(self, circuit: QuantumCircuit) -> CircuitBuilder:
        assert circuit.num_qubits == self.qubits()
        self.__circuit.data = [instr for instr in circuit.data if instr.operation.name != "initialize"]
        return self

    def gates(self, function: Callable[[QuantumCircuit], Any]) -> CircuitBuilder:
        function(self.__circuit)
        return self

    def with_data(self, data: Mapping[str, complex]) -> CircuitBuilder:
        for key, value in data.items():
            self.__data[int(key, 2)] = value
        return self

    def build(self) -> QuantumCircuit:
        self.__circuit.initialize(self.__data, range(self.qubits()))
        self.__circuit.data.insert(0, self.__circuit.data.pop())
        return self.__circuit


def _map_channel_to_amplitudes(channel: NDArray[np.uint8], w_strings: Sequence[str], h_strings: Sequence[str]) -> \
Mapping[str, complex]:
    data = {
        w_strings[x] + h_strings[y]: np.sqrt(channel[x, y], dtype=np.float64)
        for x in range(channel.shape[0])
        for y in range(channel.shape[1])
    }

    norm = np.linalg.norm(list(data.values()))

    for key, value in data.items():
        data[key] = value / norm

    return data


def _map_amplitudes_to_channel(state_vector: Sequence[complex], w_strings: Sequence[str], h_strings: Sequence[str]) -> \
NDArray[np.uint8]:
    m, n = len(w_strings), len(h_strings)
    rescale = 255 / np.max(np.power(np.abs(state_vector), 2))

    data = np.zeros((m, n), dtype=np.uint8)

    for i in range(m):
        for j in range(n):
            data[i, j] = rescale * (np.abs(state_vector[int(w_strings[i] + h_strings[j], 2)]) ** 2)

    return data


def hamming_manhattan_mapping(n_qubits: int) -> Sequence[str]:
    bit_strings = ["0", "1"]

    for _ in range(n_qubits - 1):
        pad_0 = [elem + "0" for elem in bit_strings]
        pad_1 = [elem + "1" for elem in bit_strings[::-1]]
        bit_strings = pad_0 + pad_1

    return bit_strings


def channel_to_circuit(channel: NDArray[np.uint8], pixel_mapping: PixelMapping = hamming_manhattan_mapping) -> CircuitBuilder:
    '''
    Converts a single channel into a `CircuitBuilder`. It can then be used to construct more complex circuits.
    '''
    w_qubits, h_qubits = int(np.ceil(np.log2(channel.shape[0]))), int(np.ceil(np.log2(channel.shape[1])))
    data = _map_channel_to_amplitudes(channel, pixel_mapping(w_qubits), pixel_mapping(h_qubits))

    return CircuitBuilder(w_qubits, h_qubits).with_data(data)


def image_to_circuits(image: Image, pixel_mapping: PixelMapping = hamming_manhattan_mapping) -> Generator[CircuitBuilder, None, None]:
    '''
    Takes an entire `Image` type - a tuple of 3 channels - and converts them into 3 circuits.
    '''
    for channel in image:
        yield channel_to_circuit(channel, pixel_mapping)


def sv_to_channel(state_vector: Sequence[complex], width: int, height: int,
                  pixel_mapping: PixelMapping = hamming_manhattan_mapping) -> NDArray[np.uint8]:
    '''
    Converts a state vector back into a channel.
    '''
    w_qubits, h_qubits = int(np.ceil(np.log2(width))), int(np.ceil(np.log2(height)))
    assert len(state_vector) == 2 ** (w_qubits + h_qubits)
    return _map_amplitudes_to_channel(state_vector, pixel_mapping(w_qubits), pixel_mapping(h_qubits))[:width, :height]


def run_circuit(qc: QuantumCircuit) -> Sequence[complex]:
    '''
    Runs the circuit using the `statevector_simulator` backend of Aer.
    Important: unlike in most other cases, there should be no measurements done at the end of the QC
    :return: the statevector of the circuit
    '''
    result: Result = execute(qc, Aer.get_backend('statevector_simulator')).result()
    return result.get_statevector()
