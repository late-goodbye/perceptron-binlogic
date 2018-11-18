class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.n_errors = []

    def net_input(self, X):
        """Функция чистого входа"""
        return sum([x * w for x, w in zip(X, self.weights[1:])]) + self.weights[0]

    def predict_(self, X):
        """Предсказывает, 0 или 1 будет на выходе"""
        return 1 if self.net_input(X) >= 0 else -1

    def fit(self, X, y):
        self.weights = [.0 for _ in range(len(X[0]) + 1)]
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict_(xi))
                for j in range(1, len(self.weights)):
                    self.weights[j] += update * xi[j-1]
                self.weights[0] += update
                errors += int(update != 0.0)
            self.n_errors.append(errors)

    def predict(self, X):
        return 1 if self.predict_(X) == 1 else 0


if __name__ == '__main__':
    X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    y = [-1, -1, -1, 1]

    p = Perceptron()
    p.fit(X, y)

    print('Weights:', p.weights)
    print('Errors:', p.n_errors)

    assert p.predict([0, 0]) == 0
    assert p.predict([0, 1]) == 0
    assert p.predict([1, 0]) == 0
    assert p.predict([1, 1]) == 1
