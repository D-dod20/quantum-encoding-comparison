"""
4 ways to put data into a quantum model:
- basis: turn each feature into 0s and 1s
- angle: use each feature as a rotation angle (1 qubit per feature)
- amplitude: pack all features into fewer qubits
- reupload: angle method repeated several times
"""
import math
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Ways to load data into the circuit

def basis_encode(x_bits, wires):
    """Load a list of 0s and 1s into the qubits."""
    qml.BasisEmbedding(features=x_bits, wires=wires)


def binarize(X_train, X_other, bits=3):
    """Turn each feature into 0s and 1s."""
    n_levels = 2 ** bits
    edges = [np.quantile(X_train[:, j], np.linspace(0, 1, n_levels + 1)[1:-1])
             for j in range(X_train.shape[1])]

    def to_bits(X):
        levels = np.stack([np.digitize(X[:, j], edges[j]) for j in range(X.shape[1])], axis=1)
        # Turn each number into bits
        out = [[(int(l) >> (bits - 1 - k)) & 1 for l in row for k in range(bits)] for row in levels]
        return np.array(out)

    return to_bits(X_train), to_bits(X_other)


def angle_encode(x, wires):
    """Use each feature as a rotation angle."""
    qml.AngleEmbedding(features=x, wires=wires, rotation="Y")


def amplitude_encode(x, wires):
    """Pack all features into the qubits."""
    qml.AmplitudeEmbedding(features=x, wires=wires, pad_with=0.0, normalize=True)


def reupload_encode(x, wires, weights_block):
    """Load the data, then a trainable layer, and repeat."""
    for layer_weights in weights_block:
        qml.AngleEmbedding(features=x, wires=wires, rotation="Y")
        qml.StronglyEntanglingLayers(weights=layer_weights[None, ...], wires=wires)


# The quantum model

def make_circuit(method, n_wires, n_layers, n_reupload_layers=3):
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, weights):
        wires = list(range(n_wires))
        if method == "basis":
            basis_encode(x, wires)
            qml.StronglyEntanglingLayers(weights=weights, wires=wires)
        elif method == "angle":
            angle_encode(x, wires)
            qml.StronglyEntanglingLayers(weights=weights, wires=wires)
        elif method == "amplitude":
            amplitude_encode(x, wires)
            qml.StronglyEntanglingLayers(weights=weights, wires=wires)
        elif method == "reupload":
            reupload_encode(x, wires, weights)
        else:
            raise ValueError(method)
        return qml.expval(qml.PauliZ(0))

    return circuit


def weight_shape(method, n_wires, n_layers):
    if method == "reupload":
        return (n_layers, n_wires, 3)
    return (n_layers, n_wires, 3)


def train_and_evaluate(method, X_train, y_train, X_test, y_test,
                       n_wires, n_layers=2, epochs=30, lr=0.05, seed=0):
    """Train the model and return test accuracy."""
    rng = np.random.default_rng(seed)
    circuit = make_circuit(method, n_wires, n_layers)
    shape = weight_shape(method, n_wires, n_layers)
    weights = pnp.array(rng.normal(0, 0.1, size=shape), requires_grad=True)
    y_train_pm = 2 * np.asarray(y_train) - 1  # Change labels 0/1 to -1/+1

    def predict_one(x, w):
        return circuit(x, w)

    def cost(w):
        preds = pnp.stack([predict_one(x, w) for x in X_train])
        return pnp.mean((preds - y_train_pm) ** 2)

    opt = qml.AdamOptimizer(stepsize=lr)
    for _ in range(epochs):
        weights = opt.step(cost, weights)

    scores = np.array([float(predict_one(x, weights)) for x in X_test])
    y_pred = (scores > 0).astype(int)
    return float(np.mean(y_pred == np.asarray(y_test)))


# Get the data ready for each method

def prepare(method, X_tr, X_te, n_features=4, bits=3):
    """Shrink the data with PCA and get it ready for the chosen method."""
    if method == "basis":
        pca = PCA(n_components=n_features).fit(X_tr)
        Z_tr, Z_te = pca.transform(X_tr), pca.transform(X_te)
        Xb_tr, Xb_te = binarize(Z_tr, Z_te, bits=bits)
        return Xb_tr, Xb_te, Xb_tr.shape[1]
    if method in ("angle", "reupload"):
        pca = PCA(n_components=n_features).fit(X_tr)
        Z_tr, Z_te = pca.transform(X_tr), pca.transform(X_te)
        scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(Z_tr)
        return scaler.transform(Z_tr), scaler.transform(Z_te), n_features
    if method == "amplitude":
        pca = PCA(n_components=n_features).fit(X_tr)
        Z_tr, Z_te = pca.transform(X_tr), pca.transform(X_te)
        n_wires = math.ceil(math.log2(n_features))
        return Z_tr, Z_te, n_wires
    raise ValueError(method)


if __name__ == "__main__":
    # Quick test on a small part of the data to check that it runs
    data = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.3, stratify=data.target, random_state=0)
    X_tr, y_tr = X_tr[:40], y_tr[:40]
    X_te, y_te = X_te[:20], y_te[:20]
    for method in ["basis", "angle", "amplitude", "reupload"]:
        Xtr, Xte, n_wires = prepare(method, X_tr, X_te, n_features=4, bits=2)
        acc = train_and_evaluate(method, Xtr, y_tr, Xte, y_te,
                                 n_wires=n_wires, n_layers=2, epochs=2, seed=0)
        print(f"{method:10s} smoke-test accuracy (not a result): {acc:.2f}")
