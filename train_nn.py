import numpy as np
import scipy.stats as ss
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── 1. Загрузка и препроцессинг ──────────────────────────────────────────────
print("Загружаем MNIST...")
X_all, y_all = fetch_openml('mnist_784', as_frame=True, return_X_y=True, version=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, train_size=5000, test_size=5000, random_state=1337
)

pca    = PCA(30, random_state=1337).fit(X_train)
scaler = StandardScaler().fit(pca.transform(X_train))

x      = scaler.transform(pca.transform(X_train))
y      = y_train.astype(int).values

x_test = scaler.transform(pca.transform(X_test))
y_test = y_test.astype(int).values

# ── 2. Вспомогательные функции ────────────────────────────────────────────────
def dense(x, w, b):
    return x @ w + b

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    """Производная ReLU: 1 там, где x > 0"""
    return (x > 0).astype(float)

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)   # числовая стабильность
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(probs, y):
    n = len(y)
    return -np.log(probs[np.arange(n), y] + 1e-12).mean()

def forward(x, w1, b1, w2, b2):
    z1    = dense(x, w1, b1)
    a1    = relu(z1)
    z2    = dense(a1, w2, b2)
    probs = softmax(z2)
    return z1, a1, z2, probs

def backward(x, y, z1, a1, probs, w1, b1, w2, b2):
    n = len(y)

    # Градиент лосса по z2 (до softmax)
    dz2 = probs.copy()
    dz2[np.arange(n), y] -= 1
    dz2 /= n                          # (n, 10)

    # Градиенты для второго слоя
    dw2 = a1.T @ dz2                  # (30, 10)
    db2 = dz2.sum(axis=0, keepdims=True)  # (1, 10)

    # Проброс градиента через второй Linear и ReLU
    da1 = dz2 @ w2.T                  # (n, 30)
    dz1 = da1 * relu_grad(z1)         # (n, 30)

    # Градиенты для первого слоя
    dw1 = x.T @ dz1                   # (features, 30)
    db1 = dz1.sum(axis=0, keepdims=True)  # (1, 30)

    return dw1, db1, dw2, db2

def accuracy(probs, y):
    return (probs.argmax(axis=1) == y).mean()

# ── 3. Инициализация весов ────────────────────────────────────────────────────
rng = np.random.default_rng(42)
np.random.seed(42)

w1 = ss.norm(scale=0.1).rvs(size=(x.shape[1], 30))
b1 = ss.norm(scale=0.1).rvs(size=(1, 30))
w2 = ss.norm(scale=0.1).rvs(size=(30, 10))
b2 = ss.norm(scale=0.1).rvs(size=(1, 10))

# ── 4. Обучение (mini-batch SGD) ─────────────────────────────────────────────
lr         = 0.05
n_epochs   = 200
batch_size = 256
n          = len(x)

best_acc   = 0.0
best_params = (w1.copy(), b1.copy(), w2.copy(), b2.copy())

for epoch in range(1, n_epochs + 1):
    # Перемешиваем данные
    idx = np.random.permutation(n)
    x_s, y_s = x[idx], y[idx]

    for start in range(0, n, batch_size):
        xb = x_s[start:start + batch_size]
        yb = y_s[start:start + batch_size]

        z1_, a1_, z2_, probs_ = forward(xb, w1, b1, w2, b2)
        dw1, db1, dw2, db2    = backward(xb, yb, z1_, a1_, probs_, w1, b1, w2, b2)

        w1 -= lr * dw1
        b1 -= lr * db1
        w2 -= lr * dw2
        b2 -= lr * db2

    # Оценка на трейне
    _, _, _, probs_train = forward(x, w1, b1, w2, b2)
    loss = cross_entropy_loss(probs_train, y)
    acc  = accuracy(probs_train, y)

    # Оценка на тесте
    _, _, _, probs_test = forward(x_test, w1, b1, w2, b2)
    acc_test = accuracy(probs_test, y_test)

    if acc_test > best_acc:
        best_acc = acc_test
        best_params = (w1.copy(), b1.copy(), w2.copy(), b2.copy())

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | loss={loss:.4f} | train_acc={acc:.4f} | test_acc={acc_test:.4f}")

print(f"\nЛучший test accuracy: {best_acc:.4f}")

# ── 5. Сохранение лучших весов ────────────────────────────────────────────────
w1, b1, w2, b2 = best_params

params = np.hstack([np.vstack([w1, b1]), np.vstack([w2, b2])])
with open('params.txt', 'w') as file:
    file.write('\n'.join([' '.join(map(str, row)) for row in params]))

print("Веса сохранены в params.txt")
