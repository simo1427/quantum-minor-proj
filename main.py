from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

from qmputils import retrieve_token

qc = QuantumCircuit(2)

# Provide your IBM Quantum API token here.
# These two lines only need to be run once, as your credentials get saved in $HOME/.qiskit/qiskit-ibm.json.
# See more: https://docs.quantum.ibm.com/start/setup-channel
api_token = retrieve_token()
service = QiskitRuntimeService(channel="ibm_quantum", token=api_token)

backend = service.backend("ibmq_qasm_simulator")


job = Sampler(backend).run(qc, shots=1024)
print(f"job id: {job.job_id()}")
result = job.result()
print(result)

