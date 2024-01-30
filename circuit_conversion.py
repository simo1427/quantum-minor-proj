from __future__ import annotations

import time
from typing import Callable, Dict, Mapping, Generator, Optional, Protocol, Tuple, Sequence, Union, cast

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumRegister, execute, Aer
from qiskit.circuit import QuantumCircuit
from qiskit.result import Result

from image_preprocessing import Image


PixelMapping = Callable[[int], Sequence[str]]


class Effect(Protocol):
    def __call__(self, circuit: QuantumCircuit, **kwargs) -> None:
        pass


class CircuitBuilder:
    '''
    Helper class for building circuits with multiple registers from state vectors.
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


def _qubits_from_size(size) -> int:
    '''
    Returns the amount of qubits needed to encode a channel.
    :param size: The length of the channel
    :return: The number of qubits needed
    '''
    return int(np.ceil(np.log2(size)))


def _map_channel_to_amplitudes(channel: NDArray[np.uint8], bit_strings: Sequence[str]) -> Mapping[str, complex]:
    '''
    Returns a mapping from the given list of bit-strings to the amplitudes in the channel.
    :param channel: the 2x2 color channel being mapped
    :param bit_strings: a list of bit-strings corresponding 
    :return: the mapping from bit-strings to amplitudes
    '''
    assert channel.ndim == 2, f'This method operates on one channel at a time, but the channel is:\n{channel}'

    data = {
        bit_strings[x] + bit_strings[y]: np.sqrt(channel[x, y], dtype=np.float64)   # compute bit-string for each pixel along x and y-axis
        for x, y in np.ndindex(channel.shape)
    }

    norm = np.linalg.norm(list(data.values()))

    for key, value in data.items():
        data[key] = value / norm    # normalize each value

    return data


def _map_probabilities_to_channel(probabilities: Sequence[float], max_color: int, bit_strings: Sequence[str]) -> NDArray[np.uint8]:
    '''
    Returns a channel by mapping from the given list of bit-strings to the probabilities for each pixel
    :param probabilities: the 2x2 color channel being mapped
    :param bit_strings: a list
    :return: the mapping from bit-strings to amplitudes
    '''
    n = len(bit_strings)
    norm = np.max(probabilities)

    data = np.zeros((n, n), dtype=np.uint8)

    for i in range(n):
        for j in range(n):
            data[i, j] = max_color * probabilities[int(bit_strings[i] + bit_strings[j], 2)] / norm

    return data


def manhattan_mapping(n_qubits: int) -> Sequence[str]:
    bit_strings = ["0", "1"]

    for _ in range(n_qubits - 1):
        pad_0 = [elem + "0" for elem in bit_strings]
        pad_1 = [elem + "1" for elem in bit_strings[::-1]]
        bit_strings = pad_0 + pad_1

    return bit_strings


def sequential_mapping(n_qubits: int) -> Sequence[str]:
    '''
    Creates a sequence of bit-strings in lexicographic order, used for mapping qubit states to pixels, and vice-versa.
    :param n_qubits: The number of qubits to create the mapping for
    :return: The sequence of bit-strings
    '''
    return [f"{i:0{n_qubits}b}" for i in range(2 ** n_qubits)]


def channel_to_circuit_builder(channel: NDArray[np.uint8], pixel_mapping: PixelMapping = sequential_mapping) -> CircuitBuilder:
    '''

    :param channel:
    :param pixel_mapping:
    :return:
    '''
    assert channel.ndim == 2, 'This method operates on one channel at a time, but the channel is:\n{channel}'
    n_qubits = _qubits_from_size(channel.shape[0])
    data = _map_channel_to_amplitudes(channel, pixel_mapping(n_qubits))
    return CircuitBuilder(2 * n_qubits).with_data(data)


def image_to_circuits(image: Image, max_colors: NDArray[np.uint8], pixel_mapping: PixelMapping = sequential_mapping) -> Generator[
    Tuple[int, CircuitBuilder], None, None]:

    for index, channel in enumerate(image):
        if max_colors[index] > 0:
         yield (index, channel_to_circuit_builder(channel, pixel_mapping))


def images_to_circuits(images: Sequence[Image], max_colors: NDArray[np.uint8], pixel_mapping: PixelMapping = sequential_mapping) -> \
    Generator[Tuple[int, CircuitBuilder], None, None]:
    assert all(image.shape == images[0].shape for image in images), "All images must have the same shape"

    for index, channels in enumerate(zip(*images)):
        
        if all(max_color > 0 for max_color in max_colors[index]):
            qubits = [2 * _qubits_from_size(channel.shape[0]) for channel in channels]
            data = [_map_channel_to_amplitudes(channels[i], pixel_mapping(qubits[i] // 2)) for i in range(len(channels))]

            yield (index, CircuitBuilder(*qubits).with_data(*data))


def probabilities_to_channel(probabilities: NDArray[np.float64], max_color: int, pixel_mapping: PixelMapping = sequential_mapping) -> \
Generator[NDArray[np.uint8], None, None]:
    assert probabilities.ndim == 2, f'{probabilities}'

    for register in probabilities:
        n_qubits = _qubits_from_size(len(register)) // 2
        channel = _map_probabilities_to_channel(register, max_color, pixel_mapping(n_qubits))
        yield channel


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

    for key, value in dists_or_counts.items():

        if isinstance(key, str):
            key = int(key, 2)

        for i in range(reg_count):
            index = key >> (i * reg_size)
            index &= (1 << reg_size) - 1
            probabilities[i, index] += value / norm

    return probabilities


@timer
def run_circuit(circuit: QuantumCircuit, device: str = "CPU", shots: Optional[int] = None, use_statevector: bool = False) -> NDArray[np.float64]:
    '''
    Runs the circuit using the Aer backend.
    :param circuit: The quantum circuit to execute
    :param device: The device where the simulation is supposed to be executed on, defaults to CPU
    :param shots: The number of shots the simulation should sample the result, defaults to None
    :param use_statevector: Whether or not the probabilities should be computed from a statevector, disregards shot counting when enabled
    :return: The probabilities of the circuit's state
    '''
    backend = Aer.get_backend("statevector_simulator" if use_statevector else "qasm_simulator")
    backend.set_options(device=device)

    result: Result = execute(circuit, backend, shots=shots if not use_statevector else 1).result()
    data: Union[Dict[str, int], Dict[int, float]]= {
        int(key, 2): np.abs(value) ** 2 
        for key, value in result.get_statevector().to_dict().items()
    } if use_statevector else cast(Dict[str, int], result.get_counts())

    return _extract_probabilities(data, circuit)
