from image_effects import *


def outer(i=0, j=0):
    def tmp(qc: QuantumCircuit):
        column_qubits, row_qubits = range(0, qc.num_qubits // 2), range(qc.num_qubits // 2, qc.num_qubits)
        qc.cx(qc.qubits[i], qc.qubits[j])
    return tmp


if __name__ == "__main__":
    for i in range(8):
        for j in range(8):
            if i != j:
                apply_effect_to_image(
                    'media/pattern.png',
                    f'media/cnot-qubits[{i}-{j}].png',
                    outer(i, j),
                    2_000
                )
