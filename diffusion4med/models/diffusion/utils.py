def exists(x):
    return x is not None

def extract(arr, time, shape):
    batch_size = time.shape[0]
    out = arr.gather(-1, time)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(time.device)
