from typing import Optional, Union, Callable

import cv2
import numpy as np
from apng import APNG
from qiskit import QuantumCircuit
from qiskit.extensions import UnitaryGate
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService, Batch, Sampler, Options

from timing_curves import linear
from circuit_conversion import Effect, image_to_circuits, probabilities_to_channel, \
    run_circuit, _extract_probabilities
from image_preprocessing import image_read, image_pad


# IMPORTANT: all effect functions must have the same signature of the Effect class


def rx_gates(circuit: QuantumCircuit, theta: float, **kwargs) -> None:
    """
    Applies a R_x(theta) gate on each qubit of the given quantum circuit.
    :param circuit: The circuit to be modified
    :param theta: The angle of the R_y(theta) gate
    """
    circuit.rx(theta, range(circuit.num_qubits))

    # IMPORTANT: NO IMAGE WAS OUTPUT WHEN USING PI/16!! TODO: LOOK INTO WHY


def ry_gates(circuit: QuantumCircuit, theta: float, **kwargs) -> None:
    """
    Applies a R_y(theta) gate on each qubit of the given quantum circuit.
    :param circuit: The circuit to be modified
    :param theta: The angle of the R_y(theta) gate
    """
    circuit.ry(theta, range(circuit.num_qubits))

    # IMPORTANT: A BLACK IMAGE WAS OUTPUT WHEN USING PI/8!! TODO: LOOK INTO WHY


def rz_gates(circuit: QuantumCircuit, theta: float, **kwargs) -> None:
    """
    Applies a R_z(theta) gate on each qubit of the given quantum circuit.
    :param circuit: The circuit to be modified
    :param theta: The angle of the R_z(theta) gate
    """
    circuit.rz(theta, range(circuit.num_qubits))


def single_rx_gate(circuit: QuantumCircuit, i: int, theta: float, **kwargs) -> None:
    """
    Applies a R_x(theta) gate on a specific qubit of the given quantum circuit.
    :param circuit: The circuit to be modified
    :param theta: The angle of the R_x(theta) gate
    :param i: The index of the qubit
    """
    circuit.rx(theta, i)

    # IMPORTANT: OUTPUT DIDN'T SEEM TO CHANGE FOR PI/16! TODO: LOOK INTO WHY
    # possible explanation: the effect is barely visible


def single_hadamard(circuit: QuantumCircuit, i: int, **kwargs) -> None:
    """
    Applies a H gate on a specific qubit of the given quantum circuit.
    :param circuit: The circuit to be modified
    :param i: The index of the qubit
    """
    circuit.h(i)


def some_mix_1(circuit: QuantumCircuit, **kwargs) -> None:
    """
    Applies a custom sequence of gates on the quantum circuit.
    :param circuit: The circuit to be modified
    """
    circuit.h(0)
    circuit.rx(-np.pi/3, 4)


def swap_gate(circuit: QuantumCircuit, i: int, j: int, **kwargs) -> None:
    """
    Applies a SWAP gate on two specific qubits of the given quantum circuit.
    :param circuit: The circuit to be modified
    :param i: The index of the 1st qubit
    :param j: The index of the 2nd qubit
    """
    assert i != j, "The indices should be different!"

    circuit.swap(i, j)


# partially swaps amplitudes two registers (images) based on https://en.wikipedia.org/wiki/List_of_quantum_logic_gates#Non-Clifford_swap_gates
def partial_swap(circuit: QuantumCircuit, alpha: float, i: int, j: int, on_registers: bool = False, **kwargs) -> None:
    """
    Applies a SWAP^alpha gate (partial swap gate) on two specific qubits/registers of the given quantum circuit.
    :param circuit: The circuit to be modified
    :param i: The index of the 1st qubit/register
    :param j: The index of the 2nd qubit/register
    :param on_registers: Whether or not to apply the gate onto whole registers instead of individual qubits, defaults to False
    """
    assert i != j, "The indices should be different!"

    angle = np.exp(1j * np.pi * alpha)
    gate = UnitaryGate(
        np.array([[1,               0,               0, 0],
                  [0, (1 + angle) / 2, (1 - angle) / 2, 0],
                  [0, (1 - angle) / 2, (1 + angle) / 2, 0],
                  [0,               0,               0, 1]]),
        label="SWAP^alpha gate"
    )

    circuit.append(gate, [circuit.qregs[i] if on_registers else i, circuit.qregs[j] if on_registers else j])


def apply_effect_to_image(
        filename: str,
        output_filename: str,
        effect: Effect,
        grayscale: bool = False,
        padding: str = "reflect",
        device: str = "CPU",
        shots: Optional[int] = None,
        use_statevector: bool = False,
        **kwargs
):
    """
    Modifies an image from a file using the provided effect, and saves it on the disk. When the image does not have width and/or height
    which is a power of two, it is padded by the specified padding method.
    :param filename: The name of the image file
    :param output_filename: The name of the output image file
    :param effect: The effect which should be applied
    :param greyscale: Whether or not the image should be loaded in greyscale, defaults to False
    :param padding: The padding method used when the image is not required size, defaults to 'reflect'
    :param devide: The device used in the simulation, defaults to 'CPU'
    :param shots: The number of shots for sampling the simulation or None when the simulation should decide, defaults to None
    :param use_statevector: Whether or not the simulation should use statevectors, defaults to False
    """
    image = image_pad(image_read(filename, grayscale=grayscale), padding=padding)   # Load, pad and convert the image to circuits
    max_colors = np.max(image, axis=(1, 2))

    circuit_builders = image_to_circuits(image, max_colors)

    circuits = [
        (index, builder.apply_effect(effect, **kwargs).build(measure_all=not use_statevector))
        for index, builder in circuit_builders
    ]  # Build the circuits

    channels = [[channel] for channel in image]

    for (index, circuit), max_color in zip(circuits, max_colors):
        channels[index] = list(probabilities_to_channel(run_circuit(circuit, device=device, shots=shots, use_statevector=use_statevector), max_color))

    channels = np.array(channels).squeeze(axis=1)  # Squeeze along the register axis

    cv2.imwrite(output_filename, np.stack(channels, axis=2))


def apply_effect_to_image_ibm(
        filename: str,
        output_filename: str,
        effect: Effect,
        grayscale: bool = False,
        shots: int = 1024,
        noisy: bool = True,
        **kwargs
):
    """
    This method can run noisy simulations, which give slightly more accurate results, at the cost of longer
    running times. For some reason it's still quite far from running it on actual hardware, but it's better than
    nothing.

    Since this only operates on a single frame, it's easier to do batch processing.
    """
    image = image_read(filename, grayscale=grayscale)   # Load the image
    max_colors = np.max(image, axis=(1, 2))
    circuit_builders = image_to_circuits(image)

    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = "ibmq_qasm_simulator"
    noise_model = NoiseModel.from_backend(service.backend("ibm_brisbane"))

    options = Options()
    if noisy:
        options.simulator = {"noise_model": noise_model}  # Use the noise model of the Hanoi computer
    options.execution.shots = shots
    options.optimization_level = 0
    options.resilience_level = 0

    circuits = [
        builder.apply_effect(effect, **kwargs).build()
        for builder in circuit_builders
    ]  # Build the circuits

    with Batch(service=service, backend=backend):  # Speeds up computation
        sampler = Sampler(options=options)

        jobs = sampler.run(circuits)  # Schedule the job
        probabilities = [exp["quasi_dists"] for exp in jobs.result().experiments]  # Run the job and extract the results

        channels = np.array([
            list(probabilities_to_channel(_extract_probabilities(prob, circuit), max_color))
            for prob, circuit, max_color in zip(probabilities, circuits, max_colors)
        ]).squeeze(axis=1)  # squeeze along register axis

        cv2.imwrite(output_filename, cv2.resize(np.stack(channels, axis=2), (1024, 1024), interpolation=cv2.INTER_NEAREST))


def animate_image(
        filename: str,
        output_filename: str,
        effect: Effect,
        frames: int,
        fps: int = 24,
        timing_curve: Callable[[float], float] = linear,
        grayscale: bool = False,
        padding: str = "reflect",
        device: str = "CPU",
        shots: Optional[int] = None,
        use_statevector: bool = False,
        **kwargs: Union[int, float, complex]
) -> None: 
    """
    Creates an animated image 
    """
    image = image_read(filename, grayscale=grayscale)   # Load the image
    max_colors = np.max(image, axis=(1, 2))
    circuit_builders = image_to_circuits(image_pad(image, padding=padding))
    files = []

    for i in range(frames):
        t = timing_curve(i / (frames - 1))

        circuits = [
            builder.apply_effect(effect, **{
                key: (value * t if isinstance(value, float) else value)
                for key, value in kwargs.items()
            }).build()  # Scale real parameters by the current time
            for builder in circuit_builders
        ]

        channels = np.array([
            list(probabilities_to_channel(run_circuit(circuit, device=device, shots=shots, use_statevector=use_statevector), max_color))
            for circuit, max_color in zip(circuits, max_colors)
        ]).squeeze(axis=1)

        files.append(f'media/{i}.png')
        # cv2.resize(, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f'media/{i}.png', np.stack(channels, axis=2))

    APNG.from_files(files, delay=1000//fps).save(output_filename)