from image_effects import *

if __name__ == "__main__":
    # animate_image(
    #     "media/small-checkers.png",
    #     "media/small-checkers-animated.png",
    #     frames=600,
    # )
    apply_gate_to_image(
        'media/Flower.png',
        'media/tmp.png',
        lambda qc, regs: qc.rx(np.pi / 8, qc.qubits),
    )
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
