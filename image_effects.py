from typing import Optional, Sequence, Union, Callable

import cv2
import numpy as np
from apng import APNG
from qiskit import QuantumCircuit
from qiskit.extensions import UnitaryGate
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService, Batch, Sampler, Options

from timing_curves import linear
from circuit_conversion import Effect, image_to_circuits, images_to_circuits, probabilities_to_channel, \
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


def ry_gates(circuit: QuantumCircuit, theta: float, **kwargs) -> None:
    """
    Applies a R_y(theta) gate on each qubit of the given quantum circuit.
    :param circuit: The circuit to be modified
    :param theta: The angle of the R_y(theta) gate
    """
    circuit.ry(theta, range(circuit.num_qubits))


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
) -> None:
    """
    Modifies an image from a file using the provided effect, and saves it on the disk. When the image does not have width and/or height
    which is a power of two, it is padded by the specified padding method.
    :param filename: The name of the image file
    :param output_filename: The name of the output image file
    :param effect: The effect which should be applied
    :param greyscale: Whether or not the image should be loaded in greyscale, defaults to False
    :param padding: The padding method used when the image is not required size, defaults to 'reflect'
    :param device: The device used in the simulation, defaults to 'CPU'
    :param shots: The number of shots for sampling the simulation or None when the simulation should decide, defaults to None
    :param use_statevector: Whether or not the simulation should use statevectors, defaults to False
    """
    image = image_pad(image_read(filename, grayscale=grayscale), padding=padding)   # Load and pad the image
    max_colors = np.max(image, axis=(1, 2)) # Find max colors of the image

    circuit_builders = image_to_circuits(image, max_colors) # Create circuit builders for every present channel

    circuits = [
        (index, builder.apply_effect(effect, **kwargs).build(measure_all=not use_statevector))
        for index, builder in circuit_builders
    ]  # Build the circuits

    channels = [[channel] for channel in image] # Prepare data for all channels

    # Convert circuits into channels with the appropriate max color
    for index, circuit in circuits:
        channels[index] = list(probabilities_to_channel(run_circuit(circuit, device=device, shots=shots, use_statevector=use_statevector), max_colors[index]))

    channels = np.array(channels).squeeze(axis=1)  # Squeeze along the register axis, since there is only 1 register (for 1 image)

    cv2.imwrite(output_filename, np.stack(channels, axis=2))    # Write the output image to the disk


def apply_effect_to_image_ibm(
        filename: str,
        output_filename: str,
        effect: Effect,
        grayscale: bool = False,
        shots: int = 1024,
        noisy: bool = True,
        **kwargs
) -> None:
    """
    Modifies an image from a file using the provided effect, and saves it on the disk. This method can run noisy simulations, 
    which give slightly more accurate results, at the cost of longer running times.
    Note:
    For some reason it's still quite far from running it on actual hardware, but it's better than nothing.
    Since this only operates on a single frame, it's easier to do batch processing.
    :param filename: The name of the image file
    :param output_filename: The name of the output image file
    :param effect: The effect which should be applied
    :param greyscale: Whether or not the image should be loaded in greyscale, defaults to False
    :param shots: The number of shots for sampling the simulation, defaults to 1024
    :param noisy: Whether or not the method should run a noisy simulation, defaults to True
    """
    image = image_read(filename, grayscale=grayscale)   # Load the image
    max_colors = np.max(image, axis=(1, 2)) # Find max colors of the image
    circuit_builders = image_to_circuits(image, max_colors) # Create circuit builders for every channel

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
        (index, builder.apply_effect(effect, **kwargs).build())
        for index, builder in circuit_builders
    ]  # Build the circuits

    with Batch(service=service, backend=backend):  # Speeds up computation
        sampler = Sampler(options=options)

        jobs = sampler.run(list(circuit for _, circuit in circuits))  # Schedule the job
        probabilities = [exp["quasi_dists"] for exp in jobs.result().experiments]  # Run the job and extract the results

        channels = np.array([
            list(probabilities_to_channel(_extract_probabilities(probabilities[index], circuit), max_colors[index]))
            for index, circuit in circuits
        ]).squeeze(axis=1)  # Squeeze along register axis

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
        animate_back: bool = False,
        **kwargs
) -> None: 
    """
    Creates an animated image by interpolating the effect parameters from 0 over a number of frames.
    :param filename: The name of the image file
    :param output_filename: The name of the output image file
    :param effect: The effect which should be applied
    :param frames: The number of frames the animation should have
    :param fps: The frame rate of the animation in frames per second, defaults to 24
    :param greyscale: Whether or not the image should be loaded in greyscale, defaults to False
    :param padding: The padding method used when the image is not required size, defaults to 'reflect'
    :param device: The device used in the simulation, defaults to 'CPU'
    :param shots: The number of shots for sampling the simulation or None when the simulation should decide, defaults to None
    :param use_statevector: Whether or not the simulation should use statevectors, defaults to False
    """
    image = image_read(filename, grayscale=grayscale)   # Load the image
    max_colors = np.max(image, axis=(1, 2)) # Find max colors of the image
    circuit_builders = image_to_circuits(image_pad(image, padding=padding), max_colors) # Convert the padded image to circuits

    files = []  # The files for each frame

    for frame in range(frames):
        t = timing_curve(frame / (frames - 1))  # Compute the timestep

        circuits = [
            (index, builder.apply_effect(effect, **{
                key: (value * t if isinstance(value, float) else value) # Scale real parameters by the timestep
                for key, value in kwargs.items()
            }).build(measure_all=not use_statevector))
            for index, builder in circuit_builders
        ]   # Build the circuits

        channels = [[channel] for channel in image] # Prepare data for all channels

        # Convert circuits into channels with the appropriate max color
        for index, circuit in circuits:
            channels[index] = list(probabilities_to_channel(run_circuit(circuit, device=device, shots=shots, use_statevector=use_statevector), max_colors[index]))

        channels = np.array(channels).squeeze(axis=1)   # Squeeze along the register axis

        files.append(f'media/{frame}.png')
        cv2.imwrite(f'media/{frame}.png', np.stack(channels, axis=2))

    APNG.from_files(files + (files[::-1] if animate_back else []), delay=1000//fps).save(output_filename)   # Create an animated PNG with the computed frames


def animate_images(
        start_filename: str,
        end_filename: str,
        effect: Effect,
        frames: int,
        fps: int = 24,
        timing_curve: Callable[[float], float] = linear,
        grayscale: bool = False,
        device: str = "CPU",
        shots: Optional[int] = None,
        use_statevector: bool = False,
        animate_back: bool = False,
        **kwargs
) -> None:
    """
    Creates an animation of two images, usually as transitions from one image to the other. When used for the transition, the animation on
    the end image is the same as on the start image but in reverse. Both are saved on the disk.
    :param start_filename: The name of the start image file
    :param end_filename: The name of the end image file
    :param effect: The effect which should be applied
    :param frames: The number of frames the animation should have
    :param fps: The frame rate of the animation in frames per second, defaults to 24
    :param timing_curve: The timing curve which dictates the speed of the animation, default to the linear interpolation
    :param greyscale: Whether or not the image should be loaded in greyscale, defaults to False
    :param device: The device used in the simulation, defaults to 'CPU'
    :param shots: The number of shots for sampling the simulation or None when the simulation should decide, defaults to None
    :param use_statevector: Whether or not the simulation should use statevectors, defaults to False
    :param animate_back: Whether or not to concatenate the both animations to create a looping animation, defaults to False
    """
    images = [image_read(start_filename, grayscale=grayscale), image_read(end_filename, grayscale=grayscale)]   # Load the images
    max_colors = np.moveaxis(np.max(images, axis=(2, 3)), 1, 0) # Find max colors of both images
    circuit_builders = list(images_to_circuits(images, max_colors)) # Convert the images to circuits

    files = [[] for _ in images]    # The files for each frame per image

    for frame in range(frames):
        t = timing_curve(frame / (frames - 1))  # Compute the timestep

        circuits = [
            (index, builder.apply_effect(effect, **{
                key: (value * t if isinstance(value, float) else value) # Scale real parameters by the timestep
                for key, value in kwargs.items()
            }).build(measure_all=not use_statevector))
            for index, builder in circuit_builders
        ]   # Build the circuits

        channels = np.stack([channels for channels in zip(*images)], axis=0)    # Prepare data for all channels

        # Convert circuits into channels
        for index, circuit in circuits:
            lerped_color = max_colors[index][0] + t * (max_colors[index][1] - max_colors[index][0]) # Interpolate linearly between 1st and 2nd images' max colors
            channels[index] = list(probabilities_to_channel(run_circuit(circuit, device=device, shots=shots, use_statevector=use_statevector), lerped_color))

        for i in range(channels.shape[1]):
            tmp = channels[:, i, :, :]  # Retrive the register data
            img = np.moveaxis(tmp, 0, 2)    # Interpret the data as image

            filename = f"media/{frame}-{i}.png"
            cv2.imwrite(filename, img)  # Write the frame to disk
            files[i].append(filename)
    
    # Compose the animations and write them to the disk
    APNG.from_files(files[0] + (files[1] if animate_back else []), delay=1000//fps).save(f"{start_filename[:-4]}-transition.png")
    APNG.from_files(files[1] + (files[0] if animate_back else []), delay=1000//fps).save(f"{end_filename[:-4]}-transition.png")
