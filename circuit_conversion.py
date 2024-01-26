from __future__ import annotations

import time
from typing import Callable, Dict, Mapping, Generator, Protocol, Tuple, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumRegister, execute, Aer
from qiskit.circuit import QuantumCircuit
from qiskit.result import Result

Image = Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]
PixelMapping = Callable[[int], Sequence[str]]


class Effect(Protocol):
    def __call__(self, circuit: QuantumCircuit, **kwargs) -> None:
        pass


class CircuitBuilder:
    '''
    Helper class for building circuits
    '''

    def __init__(self, *qubits: int) -> None:
        assert len(qubits) != 0

        self.__qubits: int = np.sum(qubits)
        self.__registers: Sequence[QuantumRegister] = [QuantumRegister(register) for register in qubits]
        self.__data: NDArray[np.complex128] = np.zeros((len(self.__registers), 2 ** np.max(qubits)),
                                                       dtype=np.complex128)
        self.__instructions = []

    @property
    def qubits(self) -> int:
        return self.__qubits

    def from_circuit(self, circuit: QuantumCircuit) -> CircuitBuilder:
        assert circuit.num_qubits == self.qubits

        self.__instructions = [instr for instr in circuit.data if instr.operation.name != "initialize"]

        return self

    def apply_effect(self, effect: Effect, **kwargs) -> CircuitBuilder:
        circuit = QuantumCircuit(*self.__registers)

        effect(circuit, **kwargs)
        self.__instructions = [instr for instr in circuit.data if instr.operation.name != "initialize"]

        return self

    def with_data(self, *data: Mapping[str, complex]) -> CircuitBuilder:
        assert len(data) == len(self.__registers)
        assert all(len(elem) <= 2 ** self.__registers[i].size for i, elem in enumerate(data))

        for i in range(len(data)):
            for key, value in data[i].items():
                self.__data[i, int(key, 2)] = value

        return self

    def build(self, measure_all=True) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.__registers)

        for i, register in enumerate(self.__registers):
            circuit.initialize(self.__data[i], [register])

        circuit.data.extend(self.__instructions)

        if measure_all:
            circuit.measure_all()

        return circuit


def _compute_qubits_from_size(size) -> int:
    '''
    Returns the amount of qubits needed to encode a channel.
    :param size: The length of the channel
    :return: the number of qubits needed
    '''
    return int(np.ceil(np.log2(size)))


def _map_channel_to_amplitudes(channel: NDArray[np.uint8], bit_strings: Sequence[str]) -> Mapping[str, complex]:
    '''
    Returns a mapping from the given list of bit-strings to the amplitudes in the channel
    :param channel: the 2x2 color channel being mapped
    :param bit_strings: a list
    :return: the mapping from bit-strings to amplitudes
    '''
    assert channel.ndim == 2, f'This method operates on one channel at a time, but the channel is:\n{channel}'

    data = {
        bit_strings[x] + bit_strings[y]: np.sqrt(channel[x, y], dtype=np.float64)
        for x, y in np.ndindex(channel.shape)
    }

    norm = np.linalg.norm(list(data.values()))

    for key, value in data.items():
        data[key] = value / norm

    return data


def _map_probabilities_to_channel(probabilities: Sequence[float], bit_strings: Sequence[str]) -> NDArray[np.uint8]:
    n = len(bit_strings)
    norm = np.max(probabilities)
    max_color = 255

    data = np.zeros((n, n), dtype=np.uint8)

    for i in range(n):
        for j in range(n):
            data[i, j] = max_color * probabilities[int(bit_strings[i] + bit_strings[j], 2)] / norm

    return data


def hamming_manhattan_mapping(n_qubits: int) -> Sequence[str]:
    bit_strings = ["0", "1"]

    for _ in range(n_qubits - 1):
        pad_0 = [elem + "0" for elem in bit_strings]
        pad_1 = [elem + "1" for elem in bit_strings[::-1]]
        bit_strings = pad_0 + pad_1

    return bit_strings


def ordinal_mapping(n_qubits: int) -> Sequence[str]:
    '''
    Creates a sequence of bit-strings in lexicographic order, used for mapping qubit states to pixels, and vice-versa.
    :param n_qubits: The number of qubits to create the mapping for
    :return: The sequence of bit-strings
    '''
    return [f"{i:0{n_qubits}b}" for i in range(2 ** n_qubits)]


def channel_to_circuit_builder(channel: NDArray[np.uint8], pixel_mapping: PixelMapping = ordinal_mapping) -> CircuitBuilder:
    '''

    :param channel:
    :param pixel_mapping:
    :return:
    '''
    assert channel.ndim == 2, 'This method operates on one channel at a time, but the channel is:\n{channel}'
    n_qubits = _compute_qubits_from_size(channel.shape[0])
    data = _map_channel_to_amplitudes(channel, pixel_mapping(n_qubits))
    return CircuitBuilder(2 * n_qubits).with_data(data)


def image_to_circuits(image: Image, pixel_mapping: PixelMapping = ordinal_mapping) -> Generator[
    CircuitBuilder, None, None]:
    for channel in image:
        yield channel_to_circuit_builder(channel, pixel_mapping)


def images_to_circuits(im1: Image, im2: Image, pixel_mapping: PixelMapping = ordinal_mapping):
    for ch1, ch2 in zip(im1, im2):
        assert ch1.shape == ch2.shape

        ch1_qubits = _compute_qubits_from_size(ch1.shape[0])
        ch2_qubits = _compute_qubits_from_size(ch2.shape[0])

        data1 = _map_channel_to_amplitudes(ch1, pixel_mapping(ch1_qubits))
        data2 = _map_channel_to_amplitudes(ch2, pixel_mapping(ch2_qubits))

        yield CircuitBuilder(2 * ch1_qubits, 2 * ch2_qubits).with_data(data1, data2)


def probabilities_to_channel(probabilities: NDArray[np.float64], pixel_mapping: PixelMapping = ordinal_mapping) -> \
Generator[NDArray[np.uint8], None, None]:
    assert probabilities.ndim == 2, f'{probabilities}'

    for register in probabilities:
        n_qubits = _compute_qubits_from_size(len(register)) // 2
        probs = _map_probabilities_to_channel(register, pixel_mapping(n_qubits))
        yield probs


def timer(func):
    """Print the runtime of the decorated function"""
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def _extract_probabilities(dists_or_counts: Union[Mapping[str, int], Mapping[int, float]], circuit: QuantumCircuit) -> NDArray[np.float64]:
    reg_count, reg_size = len(circuit.qregs), np.max(list(map(lambda reg: reg.size, circuit.qregs)))
    probabilities = np.zeros((reg_count, 2 ** reg_size), dtype=np.float64)

    norm = np.sum(list(dists_or_counts.values()))

    data: Dict[int, float] = {
        (int(key, 2) if isinstance(key, str) else key): value / norm
        for key, value in dists_or_counts.items()
    }

    for key, value in data.items():

        for i in range(reg_count):
            index = key >> (i * reg_size)
            index &= (1 << reg_size) - 1
            probabilities[i, index] += value

    return probabilities


@timer
def run_circuit(qc: QuantumCircuit, shots=None) -> NDArray[np.float64]:
    '''
    Runs the circuit using the `qasm_simulator` backend of Aer.
    :return: the statevector of the circuit
    '''
    result: Result = execute(qc, Aer.get_backend('qasm_simulator'), shots=shots).result()
    counts: Dict[str, int] = result.get_counts()
    return _extract_probabilities(counts, qc)


@timer
def run_circuit_statevector(qc: QuantumCircuit, device: str = 'CPU') -> NDArray[np.float64]:
    '''
    Runs the circuit using the `statevector_simulator` backend of Aer.
    :return: the statevector of the circuit
    '''

    backend = Aer.get_backend('statevector_simulator')
    backend.set_options(device=device)
    result: Result = execute(qc, backend, shots=1).result()
    # print(type(result.get_statevector()))
    statevector: NDArray = result.get_statevector()
    # print(statevector)
    probabilities: Dict[str, float] = {key: np.abs(value) ** 2 for key, value in statevector.to_dict().items()}
    return _extract_probabilities(probabilities, qc)
