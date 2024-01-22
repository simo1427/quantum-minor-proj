from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Generator, Tuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumRegister, execute, Aer
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import SamplerResult
from qiskit.result import Result
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, RuntimeJob

Image = Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]
PixelMapping = Callable[[int], Sequence[str]]

service = QiskitRuntimeService(channel="ibm_quantum")


class CircuitBuilder:

    def __init__(self, *qubits: int) -> None:
        assert len(qubits) != 0
        self.__qubits: int = np.sum(qubits)
        self.__registers: Sequence[QuantumRegister] = [QuantumRegister(register) for register in qubits]
        self.__data: NDArray[np.complex128] = np.zeros((len(self.__registers), 2 ** self.__registers[0].size),
                                                       dtype=np.complex128)
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


def image_to_circuits(image: Image, pixel_mapping: PixelMapping = ordinal_mapping) -> Generator[
    CircuitBuilder, None, None]:
    for channel in image:
        yield channel_to_circuit(channel, pixel_mapping)


def images_to_circuits(im1: Image, im2: Image, pixel_mapping: PixelMapping = ordinal_mapping):
    for ch1, ch2 in zip(im1, im2):
        assert ch1.shape == ch2.shape

        ch1_qubits = int(np.ceil(np.log2(ch1.shape[0])))
        ch2_qubits = int(np.ceil(np.log2(ch2.shape[0])))

        data1 = _map_channel_to_amplitudes(ch1, pixel_mapping(ch1_qubits))
        data2 = _map_channel_to_amplitudes(ch2, pixel_mapping(ch2_qubits))

        yield CircuitBuilder(2 * ch1_qubits, 2 * ch2_qubits).with_data([data1, data2])


def probabilities_to_channel(probabilities: NDArray[np.float64], pixel_mapping: PixelMapping = ordinal_mapping) -> \
Generator[NDArray[
    np.uint8], None, None]:
    for register in probabilities:
        n_qubits = int(np.ceil(np.log2(len(register)))) // 2
        yield _map_amplitudes_to_channel(register, pixel_mapping(n_qubits))


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


@timer
def run_circuit(qc: QuantumCircuit, shots=None) -> Sequence[float]:
    '''
    Runs the circuit using the `statevector_simulator` backend of Aer.
    Important: unlike in most other cases, there should be no measurements done at the end of the QC
    :return: the statevector of the circuit
    '''
    qc.measure_all()
    result: Result = execute(qc, Aer.get_backend('qasm_simulator'), shots=10_000).result()
    counts: Dict[str, int] = result.get_counts()
    probabilities = np.zeros((len(qc.qregs), 2 ** qc.qregs[0].size), dtype=np.float64)

    qreg_count, qreg_size = len(qc.qregs), qc.qregs[0].size
    for key, value in counts.items():
        for i in range(qreg_count):
            index = int(key[i * qreg_size:(i + 1) * qreg_size], 2)
            try:
                probabilities[i, index] += value
            except:
                probabilities[i, index] = value

    for i, row_sum in enumerate(np.sum(probabilities, axis=1)):
        probabilities[i] /= row_sum

    return probabilities


@timer
def run_circuit_ibm(qc: QuantumCircuit, shots=None) -> RuntimeJob:
    qc.measure_all()
    return Sampler(service.backend("ibmq_qasm_simulator")).run(qc, shots=100_000)


@timer
def get_simulation_results(qc: QuantumCircuit, job: RuntimeJob) -> Sequence[float]:
    quasi_probabilities: Dict[int, float] = job.result().experiments[0]["quasi_dists"]
    probabilities = np.zeros((len(qc.qregs), 2 ** qc.qregs[0].size), dtype=np.float64)

    qreg_count, qreg_size = len(qc.qregs), qc.qregs[0].size
    for key, value in quasi_probabilities.items():
        for i in range(qreg_count):
            bit_string = f"{key:0{qreg_count * qreg_size}b}"
            index = int(bit_string[i * qreg_size:(i + 1) * qreg_size], 2)
            try:
                probabilities[i, index] += value
            except:
                probabilities[i, index] = value

    return probabilities