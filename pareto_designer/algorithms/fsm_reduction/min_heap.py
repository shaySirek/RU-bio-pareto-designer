from loguru import logger
from typing import TypeVar, Hashable, Any

T = TypeVar("T", bound=Hashable)
K = TypeVar("K", bound=Any)


class MinHeap:
    """
    A min-heap data structure with eager deletion and updates.
    It manually manages the heap structure for precise control.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[K, T]] = []
        self._item_to_index: dict[T, int] = {}

    def _swap(self, i: int, j: int) -> None:
        """Helper method to swap two elements and update their indices."""
        item_i, item_j = self._heap[i][1], self._heap[j][1]
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        self._item_to_index[item_i] = j
        self._item_to_index[item_j] = i

    def _heapify_up(self, index: int) -> None:
        """Moves an item up the heap to its correct position."""
        if index == 0:
            return

        parent_index = (index - 1) // 2
        if self._heap[index][0] < self._heap[parent_index][0]:
            self._swap(index, parent_index)
            self._heapify_up(parent_index)

    def _heapify_down(self, index: int) -> None:
        """Moves an item down the heap to its correct position."""
        min_index = index
        left_child = 2 * index + 1
        right_child = 2 * index + 2
        heap_size = len(self._heap)

        if (
            left_child < heap_size
            and self._heap[left_child][0] < self._heap[min_index][0]
        ):
            min_index = left_child
        if (
            right_child < heap_size
            and self._heap[right_child][0] < self._heap[min_index][0]
        ):
            min_index = right_child

        if min_index != index:
            self._swap(index, min_index)
            self._heapify_down(min_index)

    def update(self, item: T, key: K) -> None:
        """Inserts a new item or updates an existing item's key."""
        if item in self._item_to_index:
            old_index = self._item_to_index[item]
            old_key = self._heap[old_index][0]

            if old_key == key:
                return

            self._heap[old_index] = (key, item)

            if key < old_key:
                self._heapify_up(old_index)
            else:
                self._heapify_down(old_index)
        else:
            new_index = len(self._heap)
            self._heap.append((key, item))
            self._item_to_index[item] = new_index
            self._heapify_up(new_index)

    def get_key(self, item: T) -> K:
        if item in self._item_to_index:
            index = self._item_to_index[item]
            key = self._heap[index][0]
            return key

        raise IndexError(f"Cannot get non-existent item: {item}")

    def extract_min(self) -> tuple[T, K]:
        """
        Removes and returns the item with the minimum key.
        This is the standard, efficient implementation for an eager-deletion heap.
        """
        if not self._heap:
            logger.error("Attempted to extract_min from an empty heap.")
            raise IndexError("extract_min from empty heap")

        min_key, min_item = self._heap[0]

        if len(self._heap) == 1:
            self._heap.pop()
            del self._item_to_index[min_item]
            return min_item, min_key

        last_key, last_item = self._heap.pop()
        self._heap[0] = (last_key, last_item)
        self._item_to_index[last_item] = 0
        del self._item_to_index[min_item]

        self._heapify_down(0)

        return min_item, min_key

    def delete(self, item: T) -> None:
        """Deletes a specific item and immediately fixes the heap."""
        if item not in self._item_to_index:
            logger.warning(f"Attempted to delete non-existent item: {item}.")
            return

        index_to_delete = self._item_to_index[item]
        last_item = self._heap[-1][1]

        self._swap(index_to_delete, len(self._heap) - 1)
        self._heap.pop()
        del self._item_to_index[item]

        if self._heap and item != last_item:
            self._heapify_down(index_to_delete)
            self._heapify_up(index_to_delete)

    def peek_min(self) -> tuple[T, K] | None:
        """Returns the minimum item without removing it."""
        if not self._heap:
            return None

        key, item = self._heap[0]
        return item, key

    def is_empty(self) -> bool:
        """Checks if the heap is empty."""
        return not self._heap

    def __len__(self) -> int:
        """Returns the number of items in the heap."""
        return len(self._heap)

    def __repr__(self) -> str:
        """Provides a string representation for debugging."""
        return f"MinHeap(size={self.__len__()})"

    def __eq__(self, value):
        if hasattr(value, "__len__"):
            return sorted(value) == sorted(self._heap)

        return False


if __name__ == "__main__":
    logger.info("Starting MinHeap demonstration.")
    min_heap = MinHeap()

    # Insert items
    min_heap.update("task1", 10)
    min_heap.update("task2", 5)
    logger.info(f"Min heap after initial insertions: {min_heap!r}")

    # Update a key
    min_heap.update("task1", 3)
    logger.info(f"Min heap after updating 'task1' key: {min_heap!r}")

    # Extract minimum
    extracted_item = min_heap.extract_min()
    assert extracted_item == ("task1", 3)
    logger.info(f"Min heap after extracting min: {min_heap!r}")

    # Peek min
    peeked_item = min_heap.peek_min()
    assert peeked_item == ("task2", 5)
    logger.info(f"Peeked item: {peeked_item}")

    # Delete an item
    min_heap.delete("task2")
    logger.info(f"Min heap after deleting 'task2': {min_heap!r}")

    assert min_heap.is_empty()
    logger.info("Min heap is empty")
