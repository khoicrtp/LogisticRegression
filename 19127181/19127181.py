import numpy as np
import copy
import json

data = np.loadtxt(open("training_data.txt", "r"), delimiter=",")
X = data[:, 0:2]
y = data[:, 2]


def map_feature(x1, x2):
    degree = 6
    out = np.ones([len(x1), int((degree + 1) * (degree + 2) / 2)])
    idx = 1

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            a1 = x1 ** (i - j)
            a2 = x2 ** j
            out[:, idx] = a1 * a2
            idx += 1

    return out


def map_feature2(x1, x2):
    degree = 6
    out = np.ones([1, int((degree + 1) * (degree + 2) / 2)])
    idx = 1

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            a1 = x1 ** (i - j)
            a2 = x2 ** j
            out[:, idx] = a1 * a2
            idx += 1

    return out

def sigmoid(z):
    u = np.array(z, dtype=np.float128)
    return 1 / (1 + np.exp(-u))


def regularize_theta(theta, lamb):
    sum = 0
    for i in range(1, len(theta)):
        sum += theta[i] ** 2.0
    return 1.0*lamb / (2.0*m) * sum


def compute_cost(theta, X, y, lamb):
    J = 1.0 / m * (np.dot(-y.T, np.log(sigmoid(X.dot(theta))))
                   - np.dot((1 - y).T, np.log(1 - sigmoid(X.dot(theta))))) \
        + regularize_theta(theta, lamb)
    return J


def compute_gradient(theta, X, y, lamb):
    indexArr = np.eye(len(theta))
    indexArr[0, 0] = 0
    grad = 1.0 / m * np.dot(
        (sigmoid(X.dot(theta)) - y).T, X).T \
        + 1.0 * lamb / m * (indexArr.dot(theta))
    return grad


def initTheta(value, n):
    res = []
    for i in range(n):
        res.append(value)
    return res


def gradient_descent(theta, alpha, X, y, lamb, maxStep=10000):
    while maxStep > 0:  # True:
        gradientVector = compute_gradient(theta, X, y, lamb)
        temp = copy.deepcopy(theta)
        for i in range(0, len(theta)):
            theta[i] = theta[i] - alpha * gradientVector[i]
        prevCost = compute_cost(temp, X, y, lamb)
        aftCost = compute_cost(theta, X, y, lamb)
        if(abs(prevCost - aftCost) < 0.000001):
            return theta
        maxStep -= 1
    return theta


def validate(theta, X):
    p = sigmoid(X.dot(theta))
    for i in range(len(p)):
        if(p[i] >= 0.5):
            p[i] = 1
        else:
            p[i] = 0
    return p


def predict(theta, a):
    formal=map_feature2(a[0], a[1])
    p = sigmoid(formal.dot(theta))
    if p>=0.5:
        return True
    else:
        return False


def percent(p, y):
    count = 0
    for i in range(len(p)):
        if p[i] == y[i]:
            count += 1
    return count/len(p)*100


def toString(a):
    res = ""
    for i in range(len(a)-1):
        res += str(a[i])+","
    res+=str(a[len(a)-1])
    return res


if __name__ == '__main__':
    with open('config.json',) as f:
        data = json.load(f)

    X = map_feature(X[:, 0], X[:, 1])
    m, n = X.shape
    theta = initTheta(1, n)
    lamb = data["Lambda"]
    alpha = data["Alpha"]
    numIter = data["NumIter"]
    print(lamb, alpha, numIter)
    print("PREV COST:", compute_cost(theta, X, y, lamb))
    print("GRAD:", compute_gradient(theta, X, y, lamb))

    newTheta = gradient_descent(theta, alpha, X, y, lamb, numIter)
    print("NEW THETA:", newTheta)
    aftCost = compute_cost(newTheta, X, y, lamb)
    print("AFTER COST:", aftCost)
    aftGrad = compute_gradient(newTheta, X, y, lamb)
    print("GRAD:", aftGrad)
    p = validate(newTheta, X)

    with open('model.json', 'w') as fout:
        dict_obj = {"Theta": toString(newTheta)}
        json_object = json.dumps(dict_obj)
        fout.write(json_object)
    print("ACCURACY:", percent(p, y))

    with open('accuracy.json', 'w') as fout2:
        dict_obj = {"Accuracy": str(percent(p, y))}
        json_object = json.dumps(dict_obj)
        fout2.write(json_object)

    test=[0.18,0.93]
    print("PREDICT: ", predict(newTheta, test))

    test=[-0.44,0.2]
    print("PREDICT: ", predict(newTheta, test))
