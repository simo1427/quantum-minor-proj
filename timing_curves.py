import numpy as np

def linear(t: float) -> float:
    return t

def easeInSine(t: float) -> float:
    return 1 - np.cos(np.pi * t / 2)

def easeOutSine(t: float) -> float:
    return np.sin(np.pi * t / 2)

def easeInOutSine(t: float) -> float:
    return (1 - np.cos(np.pi * t)) / 2

def easeInCubic(t: float) -> float:
    return t ** 3

def easeOutCubic(t: float) -> float:
    return 1 - (1 - t) ** 3

def easeInOutCubic(t: float) -> float:
    return 4 * t ** 3 if (t < 0.5) else (1 - (2 * (1 - t)) ** 3) / 2

def easeInBack(t: float) -> float:
    return 2.70158 * t ** 3 - 1.70158 * t ** 2

def easeOutBack(t: float) -> float:
    return 1 + 2.70158 * (1 - t) ** 3 + 1.70158 * (1 - t) ** 2

def easeInOutBack(t: float) -> float:
    a1 = 1.70158
    a2 = a1 * 1.525
    if t < 0.5:
        return ((2 * t) ** 2 * ((a2 + 1) * 2 * t - a2)) / 2
    else:
        return ((2 * (t - 1)) ** 2 * ((a2 + 1) * 2 * (t - 1) + a2) + 2) / 2

def easeInElastic(t: float) -> float:
    if t == 0:
        return 0
    elif t == 1:
        return 1
    else:
        return -(2 ** (10 * (t - 1))) * np.sin(2 * np.pi / 3 * (10 * t - 10.75))

def easeOutElastic(t: float) -> float:
    if t == 0:
        return 0
    elif t == 1:
        return 1
    else:
        return 2 ** (-10 * t) * np.sin(2 * np.pi / 3 * (10 * t - .75)) + 1

def easeInOutElastic(t: float) -> float:
    if t == 0:
        return 0
    elif t == 1:
        return 1
    elif t < 0.5:
        return (-(2 ** (20 * t - 10)) * np.sin(2 * np.pi / 4.5 * (20 * t - 11.125))) / 2
    else:
        return (2 ** (-20 * t + 10) * np.sin(2 * np.pi / 4.5 * (20 * t - 11.125))) / 2 + 1