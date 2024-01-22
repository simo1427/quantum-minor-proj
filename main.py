from typing import List

from matplotlib import pyplot as plt
from qiskit import execute, Aer

from circuit_conversion import images_to_circuits
from image_effects import *


def partial_swap(percentage: float):
    def inner(qc: QuantumCircuit, regs: List[QuantumRegister]):
        qc.rxx(np.pi / 2 * percentage, regs[0], regs[1])
        qc.ryy(np.pi / 2 * percentage, regs[0], regs[1])
        qc.rzz(np.pi / 2 * percentage, regs[0], regs[1])
    return inner


def test():
    builders = [circ for circ in images_to_circuits(image_read('media/ocean.png'), image_read('media/grass.png'))]

    files = []
    for frame in range(10):

        circuits = [b.gates(partial_swap(frame / 10)).build() for b in builders]

        probabilities = [run_circuit(qc) for qc in circuits]
        channels = np.array([list(probabilities_to_channel(prob)) for prob in probabilities])

        for i in range(channels.shape[1]):
            tmp = channels[:, i, :, :]
            img = np.moveaxis(tmp, 0, 2)
            cv2.imwrite(f'media/{frame}-{i}.png', cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST))
        files.append(f'media/{frame}-1.png')
    APNG.from_files(files + files[::-1], delay=1000//10).save('media/testtest.png')


if __name__ == "__main__":
    test()
    # regs = [QuantumRegister(2), QuantumRegister(2)]
    # qc = QuantumCircuit(*regs)
    # qc.measure_all()
    #
    # result = execute(qc, Aer.get_backend('qasm_simulator'), shots=4 ** qc.num_qubits).result()
    # counts = result.get_counts()
    # print(counts)

    # animate_image(
    #     "media/checkers.png",
    #     "media/checkers-animated.png",
    #     frames=60,
    # )

    # apply_gate_to_image(
    #     'media/checkers.png',
    #     'media/checkers-animated.png',
    #     lambda qc, regs: qc.rx(np.pi / 2, qc.qubits)
    # )
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
