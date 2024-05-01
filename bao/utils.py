import time

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds to run")
        return result
    return wrapper

def is_binary(precision):
    return precision.endswith("binary")

def is_float32(precision):
    return precision.endswith("float32") 

def is_scalar(precision):
    return precision.endswith("int8") 