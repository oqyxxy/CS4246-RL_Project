import operator


class SegmentTree(object):
    """
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """
    def __init__(self, size, operation, neutral_element):

        #powers of two have no bits in common with the previous integer
        assert size > 0 and size & (size - 1) == 0, "size must be positive and a power of 2"
        self._size = size

        #a segment tree has (2*n)-1 total nodes
        self._value = [neutral_element for _ in range(2 * size)]

        self._operation = operation

        self.next_index = 0

    def reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self.reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self.reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self.reduce_helper(start, mid, 2 * node, node_start, mid),
                    self.reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._size
        if end < 0:
            end += self._size
        end -= 1
        return self.reduce_helper(start, end, 1, 0, self._size - 1)

    def set_item(self, idx, val):
        # index of the leaf
        idx += self._size
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def get_item(self, idx):
        assert 0 <= idx < self._size
        return self._value[self._size + idx]

class SumSegmentTree(SegmentTree):
    """
    SumTree allows us to sum priorities of transitions in order to assign each a probability of being sampled.
    """
    def __init__(self, size):
        super(SumSegmentTree, self).__init__(
            size=size,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        prefixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._size:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._size

class MinSegmentTree(SegmentTree):
    """
    In PrioritizedMemory, we normalize importance weights according to the maximum weight in the buffer.
    This is determined by the minimum transition priority. This MinTree provides an efficient way to
    calculate that.
    """
    def __init__(self, size):
        super(MinSegmentTree, self).__init__(
            size=size,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)