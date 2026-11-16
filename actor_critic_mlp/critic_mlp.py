"""Crítico (función de valor v̂) con una red neuronal MLP en PyTorch.

Sustituye a la línea base lineal v̂(s, w) = w · x(s). La política (theta) se
queda en NumPy; solo el crítico usa PyTorch.

Actualización (semi-gradiente, Sutton & Barto, cap. 9 y 13):

    delta <- objetivo - v̂(S, w)
    w     <- w + alpha_w * delta * grad v̂(S, w)

donde el objetivo es:
    - Actor-crítico (un paso):  R + gamma * v̂(S', w)   (0 si S' es terminal)
    - REINFORCE con línea base: G_t

El objetivo se calcula SIN gradiente: solo se deriva v̂(S, w), no v̂(S', w).
Con SGD y lr = alpha_w, minimizar 0.5 * (objetivo - v̂(S))^2 da exactamente
la actualización de arriba, porque su gradiente es -(objetivo - v̂(S)) * grad v̂(S).
"""

import numpy as np
import torch
import torch.nn as nn


class MLPCritic:
    """Función de valor v̂(s, w) aproximada con un perceptrón multicapa.

    Parámetros
    ----------
    n_features : int
        Tamaño del vector de entrada x(s) (el mismo que usaba la línea base
        lineal: estado + sesgo, y + tiempo restante si se usa).
    hidden_sizes : tuple[int]
        Neuronas por capa oculta. (64, 64) = dos capas de 64.
    activation : str
        "relu" o "tanh".
    alpha_w : float
        Tamaño de paso (learning rate) del crítico.
    optimizer : str
        "sgd" (igual que las diapositivas) o "adam" (suele ser más estable).
    zero_init_output : bool
        Si True, la capa de salida empieza en 0, así v̂(s) = 0 al inicio,
        igual que w = 0 en la versión lineal.
    max_grad_norm : float o None
        Si se da, recorta la norma del gradiente para evitar pasos enormes
        (protección contra divergencia con aproximación no lineal).
    seed : int o None
        Semilla de PyTorch para que la inicialización sea reproducible.
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes=(64, 64),
        activation: str = "relu",
        alpha_w: float = 1e-3,
        optimizer: str = "sgd",
        zero_init_output: bool = True,
        max_grad_norm=None,
        seed=None,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        # ---- Arquitectura: x -> [Linear -> act] * k -> Linear -> v̂ ----
        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), act()]
            in_size = h
        output = nn.Linear(in_size, 1)
        if zero_init_output:
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        layers.append(output)

        self.net = nn.Sequential(*layers)
        self.max_grad_norm = max_grad_norm

        if optimizer == "sgd":
            self.opt = torch.optim.SGD(self.net.parameters(), lr=alpha_w)
        elif optimizer == "adam":
            self.opt = torch.optim.Adam(self.net.parameters(), lr=alpha_w)
        else:
            raise ValueError("optimizer debe ser 'sgd' o 'adam'")

    # ------------------------------------------------------------------
    @staticmethod
    def _to_tensor(x):
        """Convierte un vector de NumPy (o lista) en tensor float32 de forma (1, n)."""
        return torch.as_tensor(np.asarray(x, dtype=np.float32)).unsqueeze(0)

    def value(self, x) -> float:
        """v̂(s, w) como número de Python, sin gradiente."""
        with torch.no_grad():
            return float(self.net(self._to_tensor(x)))

    # ------------------------------------------------------------------
    def update(self, x, target: float) -> float:
        """Un paso de semi-gradiente hacia `target`.

        Calcula delta = target - v̂(x) con el w ACTUAL, actualiza w y
        devuelve ese delta (para usarlo en la actualización de theta).
        """
        v = self.net(self._to_tensor(x)).squeeze()      # v̂(S, w), con gradiente
        delta = float(target) - float(v.detach())        # delta con el w actual

        loss = 0.5 * (float(target) - v) ** 2            # target es constante
        self.opt.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.opt.step()

        return delta

    def td_update(self, x, reward: float, next_x, done: bool, gamma: float) -> float:
        """Actor-crítico de un paso:  objetivo = R + gamma * v̂(S', w).

        Si S' es terminal, v̂(S', w) = 0. Devuelve delta.
        """
        v_next = 0.0 if done else self.value(next_x)     # sin gradiente
        target = reward + gamma * v_next
        return self.update(x, target)

    # ------------------------------------------------------------------
    def state_dict(self):
        """Pesos de la red (para copiar el mejor crítico, como w.copy())."""
        return {k: v.clone() for k, v in self.net.state_dict().items()}

    def load_state_dict(self, state):
        self.net.load_state_dict(state)

    def save(self, path: str):
        """Guarda los pesos en un archivo .pt."""
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path))


# --------------------------------------------------------------------------
# Prueba rápida (sin SUMO): python critic_mlp.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 1) Regresión: ¿aprende una función NO lineal de las features?
    critic = MLPCritic(n_features=3, hidden_sizes=(32, 32), alpha_w=1e-2,
                       optimizer="adam", seed=0)
    f = lambda x: -3.0 * x[0] ** 2 + np.sin(4 * x[1]) - x[2]
    for _ in range(5000):
        x = rng.uniform(0, 1, 3)
        critic.update(x, f(x))
    test = rng.uniform(0, 1, (200, 3))
    err = np.mean([abs(critic.value(x) - f(x)) for x in test])
    print(f"Regresión no lineal | error absoluto medio: {err:.3f}")

    # 2) TD(0): cadena de 5 estados, recompensa -1 por paso, gamma = 0.9.
    #    Valor verdadero desde el estado k (faltan 5-k pasos): -(1 - 0.9^(5-k)) / 0.1
    critic = MLPCritic(n_features=6, hidden_sizes=(32,), alpha_w=5e-2, seed=0)
    one_hot = lambda k: np.eye(6)[k]
    for _ in range(2000):
        for k in range(5):
            critic.td_update(one_hot(k), -1.0, one_hot(k + 1), done=(k == 4), gamma=0.9)
    for k in range(5):
        true_v = -(1 - 0.9 ** (5 - k)) / 0.1
        print(f"estado {k}: v̂ = {critic.value(one_hot(k)):6.3f} | verdadero = {true_v:6.3f}")