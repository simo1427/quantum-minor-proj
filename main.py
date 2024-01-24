from animation_curves import easeInOutElastic, easeOutCubic

from image_effects import *


if __name__ == "__main__":
    animate_image('media/ocean.png', 'media/result.png', frames=90, fps=30, animation_curve=easeOutCubic, shots=10_000)
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
