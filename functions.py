import numpy as np

class function:
    def __init__(self, f, gradient, dom_size):
        self.f = f
        self.gradient = gradient
        self.dom_size = dom_size # If f(x) is defined on R^p, dom_size = p
    
    def f(self, x):
        return self.f(x)
    
    def gradient(self, x):
        return self.gradient(x)
    
    
if __name__ == "__main__":
    pass