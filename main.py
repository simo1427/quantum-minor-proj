import numpy as np
from apng import APNG
from qiskit import QuantumCircuit

from circuit_conversion import channel_to_circuit, run_circuit, sv_to_channel
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
    circuit_builder = channel_to_circuit(image_read(filename, True))

    # Animate the transformation. Still a WIP
    files = []
    for i in range(1, 101):
        circuit = circuit_builder.gates(rotate_by_angle(2 * np.pi * i / 100)).build()

        channels = sv_to_channel(run_circuit(circuit), 16, 16)

        files.append(f'{i}.png')
        PilImage.fromarray(channels).convert('L').save(f'media/{i}.png')
    APNG.from_files(files, delay=50).save('media/result.png')

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

