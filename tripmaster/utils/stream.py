"""
utilities for streams
"""
import itertools
import collections

def isolate_iterators(iterable, n):
    deques = [collections.deque() for _ in range(n)]

    def new_generator(i):

        def gen(mydeque):
            while True:
                if not mydeque:             # when the local deque is empty
                    try:
                        newval = next(iterable)       # fetch a new value and
                    except StopIteration as e:
                        break 
                    
                    for j, d in enumerate(deques):        # load it to all the deques
                        d.append(newval[j])
                yield mydeque.popleft()

        return gen(deques[i])

    return [new_generator(i) for i in range(n)]


def isolate_iterators_tee(iterator_of_list, n):
    """

    Args:
        iterator_of_list:
        n:

    Returns:

    """

    duplicated_iterators = itertools.tee(iterator_of_list, n)

    def build_gen(i):
        return (x[i] for x in duplicated_iterators[i])

    isolated_iterators = [build_gen(i) for i in range(n)]

    return isolated_iterators

