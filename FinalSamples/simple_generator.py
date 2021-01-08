'''simple generator example
'''

def simple_generator(k):
    """Simple generator example
    """
    i = 0
    while i < k:
        yield i 
        print("executed after yield")
        i += 1


x = simple_generator(2)
print(next(x))
print(next(x))
print(next(x))
elems = [elem for elem in simple_generator(3)]
