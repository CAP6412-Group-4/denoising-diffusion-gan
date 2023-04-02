import timeit 
from sample.generate import main, setup_params, generate_sample

args = setup_params()

t = timeit.Timer(lambda: generate_sample(*args))
iter = 3

avg = t.timeit(iter)/iter
print(f'Average time per sample: {avg}')