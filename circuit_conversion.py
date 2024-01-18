from __future__ import annotations
from typing import Any, Callable, Dict, List, Mapping, Generator, Tuple, Sequence
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumRegister, execute, Aer
from qiskit.circuit import QuantumCircuit
from qiskit.result import Result

Image = Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]
PixelMapping = Callable[[int], Sequence[str]]


class CircuitBuilder:

    def __init__(self, *qubits: int) -> None:
        assert len(qubits) != 0
        self.__qubits: int = np.sum(qubits)
        self.__registers: Sequence[QuantumRegister] = [QuantumRegister(register) for register in qubits]
        self.__data: NDArray[np.complex128] = np.zeros((len(self.__registers), 2 ** self.qubits), dtype=np.complex128)
        self.__instructions = []

    @property
    def qubits(self) -> int:
        return self.__qubits

    def from_circuit(self, circuit: QuantumCircuit) -> CircuitBuilder:
        assert circuit.num_qubits == self.qubits
        self.__instructions = [instr for instr in circuit.data if instr.operation.name != "initialize"]
        return self

    def gates(self, function: Callable[[QuantumCircuit, Sequence[QuantumRegister]], Any]) -> CircuitBuilder:
        circuit = QuantumCircuit(*self.__registers)
        function(circuit, self.__registers)
        self.__instructions = [instr for instr in circuit.data if instr.operation.name != "initialize"]
        return self

    def with_data(self, data: Sequence[Mapping[str, complex]]) -> CircuitBuilder:
        for i in range(len(data)):
            for key, value in data[i].items():
                self.__data[i, int(key, 2)] = value
        return self

    def build(self) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.__registers)
        for i, register in enumerate(self.__registers):
            circuit.initialize(self.__data[i], [register])
        circuit.data.extend(self.__instructions)
        return circuit


def _map_channel_to_amplitudes(channel: NDArray[np.uint8], bit_strings: Sequence[str]) -> Mapping[str, complex]:
    data = {
        bit_strings[x] + bit_strings[y]: np.sqrt(channel[x, y], dtype=np.float64)
        for x in range(channel.shape[0])
        for y in range(channel.shape[1])
    }

    norm = np.linalg.norm(list(data.values()))

    for key, value in data.items():
        data[key] = value / norm

    return data


def _map_amplitudes_to_channel(probabilities: Sequence[float], bit_strings: Sequence[str]) -> NDArray[np.uint8]:
    n = len(bit_strings)
    rescale = 255 / np.max(probabilities)

    data = np.zeros((n, n), dtype=np.uint8)

    for i in range(n):
        for j in range(n):
            data[i, j] = rescale * (probabilities[int(bit_strings[i] + bit_strings[j], 2)])

    return data


def hamming_manhattan_mapping(n_qubits: int) -> Sequence[str]:
    bit_strings = ["0", "1"]

    for _ in range(n_qubits - 1):
        pad_0 = [elem + "0" for elem in bit_strings]
        pad_1 = [elem + "1" for elem in bit_strings[::-1]]
        bit_strings = pad_0 + pad_1

    return bit_strings


def ordinal_mapping(n_qubits: int) -> Sequence[str]:
    return [f"{i:0{n_qubits}b}" for i in range(2 ** n_qubits)]


def channel_to_circuit(channel: NDArray[np.uint8], pixel_mapping: PixelMapping = ordinal_mapping) -> CircuitBuilder:
    n_qubits = int(np.ceil(np.log2(channel.shape[0])))
    data = _map_channel_to_amplitudes(channel, pixel_mapping(n_qubits))
    return CircuitBuilder(2 * n_qubits).with_data([data])


def image_to_circuits(image: Image, pixel_mapping: PixelMapping = ordinal_mapping) -> Generator[CircuitBuilder, None, None]:
    for channel in image:
        yield channel_to_circuit(channel, pixel_mapping)


def probabilities_to_channel(probabilities: Sequence[float], pixel_mapping: PixelMapping = ordinal_mapping) -> NDArray[
    np.uint8]:
    n_qubits = int(np.ceil(np.log2(len(probabilities)))) // 2
    return _map_amplitudes_to_channel(probabilities, pixel_mapping(n_qubits))


def run_circuit(qc: QuantumCircuit, shots=None) -> Sequence[float]:
    '''
    Runs the circuit using the `statevector_simulator` backend of Aer.
    Important: unlike in most other cases, there should be no measurements done at the end of the QC
    :return: the statevector of the circuit
    '''
    qc.measure_all()
    result: Result = execute(qc, Aer.get_backend('qasm_simulator'), shots=4 ** qc.num_qubits).result()
    counts: Dict[str, int] = result.get_counts()
    probabilities = np.zeros((2 ** qc.num_qubits,), dtype=np.float64)
    for key, value in counts.items():
        probabilities[int(key, 2)] = value
    probabilities /= np.sum(probabilities)
    return probabilities
