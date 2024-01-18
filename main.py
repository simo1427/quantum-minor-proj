import cv2
import numpy as np
from apng import APNG
from qiskit import QuantumCircuit, Aer
from qiskit_ibm_runtime import QiskitRuntimeService

from circuit_conversion import channel_to_circuit, run_circuit, probabilities_to_channel, image_to_circuits
from image_preprocessing import image_read
from PIL import Image as PilImage


def rotate_by_angle(phi):
    '''
    Having this wrapper that returns the actual gate sequence allows us to change the phi angle dynamically.
    Change the inner method as needed.
    :param phi: the angle in radians
    '''
    def inner_method(qc: QuantumCircuit):
        qc.rx(phi, qc.qubits)
    return inner_method


def animate_image(filename: str):
    cb1, cb2, cb3 = tuple([circ for circ in image_to_circuits(image_read(filename))])

    # Animate the transformation. Still a WIP
    files = []
    frames = 60
    for i in range(frames):
        print(i)
        print('Building circuit')
        circuit1 = cb1.gates(rotate_by_angle(2 * np.pi * i / (frames - 1))).build()
        circuit2 = cb2.gates(rotate_by_angle(2 * np.pi * i / (frames - 1))).build()
        circuit3 = cb3.gates(rotate_by_angle(2 * np.pi * i / (frames - 1))).build()
        print('Getting channels')
        channels1 = probabilities_to_channel(run_circuit(circuit1))
        channels2 = probabilities_to_channel(run_circuit(circuit2))
        channels3 = probabilities_to_channel(run_circuit(circuit3))
        print('Saving image')
        files.append(f'media/{i}.png')
        cv2.imwrite(f'media/{i}.png', np.stack([channels1, channels2, channels3], axis=2))
    APNG.from_files(files, delay=1000//30).save('media/result.png')

if __name__ == '__main__':
    animate_image('media/Flower.png')


    # ================   Not needed for now   ================
    #
    #
    # qc = QuantumCircuit(2)
    #
    # # Provide your IBM Quantum API token here.
    # # These two lines only need to be run once, as your credentials get saved in $HOME/.qiskit/qiskit-ibm.json.
    # # See more: https://docs.quantum.ibm.com/start/setup-channel
    # api_token = retrieve_token()
    # service = QiskitRuntimeService(channel="ibm_quantum", token=api_token)
    #
    # backend = service.backend("ibmq_qasm_simulator")
    #
    #
    # job = Sampler(backend).run(qc, shots=1024)
    # print(f"job id: {job.job_id()}")
    # result = job.result()
    # print(result)

