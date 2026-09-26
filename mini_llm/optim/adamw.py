from ..backend import xp


class AdamW:
    def __init__(
        self, parameters, lr=3e-4, beta1=0.9, beta2=0.95,
        eps=1e-8, weight_decay=0.1
    ):
        self.parameters = list(parameters)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.step_index = 0
        self.m = [xp.zeros_like(p.data) for p in self.parameters]
        self.v = [xp.zeros_like(p.data) for p in self.parameters]

    def step(self, lr=None):
        self.step_index += 1
        eta = self.lr if lr is None else float(lr)
        c1 = 1.0 - self.beta1 ** self.step_index
        c2 = 1.0 - self.beta2 ** self.step_index

        for i, p in enumerate(self.parameters):
            g = p.grad
            self.m[i] *= self.beta1
            self.m[i] += (1.0 - self.beta1) * g
            self.v[i] *= self.beta2
            self.v[i] += (1.0 - self.beta2) * (g * g)

            m_hat = self.m[i] / c1
            v_hat = self.v[i] / c2

            if p.decay and self.weight_decay != 0.0:
                p.data *= (1.0 - eta * self.weight_decay)

            p.data -= eta * m_hat / (xp.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()
