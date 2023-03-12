def no_schedule(step: int) -> float:
    return 1.0


def zero_schedule(step: int) -> float:
    return 0.0


def mix_schedule(step: int) -> float:
    if step < 5000:
        return 1.0
    return max(0, 1 - (step - 5000) / 15000)
