# fibonacci_class.py
class Fibonacci:
    def __init__(self):
        self.cache= [0, 1]
    def __call__(self, n):
        if not (isinstance(n, int) and n>= 0): # Validate the value of n
            raise ValueError(f'Positive integer number expected, got "{n}"')
        if n< len(self.cache): # Check for computed Fibonacci numbers
            return self.cache[n]
        else:
            fib_number= self(n-1)+ self(n- 2) # Compute and cache the requested Fibonacci number
            self.cache.append(fib_number)
        return self.cache[n]
    




