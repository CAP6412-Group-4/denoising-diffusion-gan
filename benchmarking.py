import timeit 
from sample.generate import main, setup_params, generate_sample

args = setup_params()

t = timeit.Timer(lambda: generate_sample(*args))
iter = 3
print(t.timeit(iter)/iter)