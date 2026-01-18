from typing import TypeVar, Hashable, Generic

T = TypeVar("T", bound=Hashable)


class UnionFind(Generic[T]):
    """
    A Union-Find (Disjoint Set) data structure with union by rank and path compression.
    Supports adding elements, unifying sets, retrieving sets, and handling deletions.
    """

    def __init__(self):
        """Initializes an empty Union-Find data structure."""
        self._parent: dict[T, T] = {}
        self._rank: dict[T, int] = {}
        self._deleted: set[T] = set()
        self._members: dict[T, set[T]] = {}

    # Functions of standard Union-Find data structure: MakeSet, Union, Find
    def add(self, x: T) -> None:
        """Adds a singleton set containing element x."""
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
            self._members[x] = {x}

    def union(self, x: T, y: T) -> None:
        """Merges the sets containing x and y."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # Union by rank
        if self._rank[root_x] > self._rank[root_y]:
            self._parent[root_y] = root_x
            self._members[root_x].update(self._members[root_y])
            del self._members[root_y]
        elif self._rank[root_x] < self._rank[root_y]:
            self._parent[root_x] = root_y
            self._members[root_y].update(self._members[root_x])
            del self._members[root_x]
        else:
            self._parent[root_y] = root_x
            self._rank[root_x] += 1
            self._members[root_x].update(self._members[root_y])
            del self._members[root_y]

    def find(self, x: T) -> T:
        """Finds the representative of the set containing x."""
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # Path compression
        return self._parent[x]

    # Extended: Get, Remove
    def get(self, x: T) -> tuple[T, set[T]]:
        """Returns the representative of the set containing x and the set itself."""
        root = self.find(x)
        return root, self._members[root].difference(self._deleted)

    def remove(self, x: T) -> None:
        """Removes x from the data structure (marks as deleted)."""
        self._deleted.add(x)

    # Iterator that yields the element in each singleton set
    # and the origin set of elements that it represents
    def get_singleton_sets_iterator(self):
        """Yields pairs of element, `x`, and its represented elements,
        which were unified with `x` and removed.
        """
        all_items = self.get_items()
        for x in all_items:
            root = self.find(x)
            x_represented = self._members[root]
            assert x_represented.difference(self._deleted) == {x}
            yield x, x_represented

    def get_partitioning(self) -> list[list[T]]:
        all_items = self.get_items()
        return [
            list(self._members[root].difference(self._deleted))
            for root in set(self.find(x) for x in all_items)
        ]

    def get_items(self) -> set[T]:
        return set(self._parent.keys()).difference(self._deleted)


if __name__ == "__main__":
    uf = UnionFind[str]()
    uf.add("a")
    uf.add("b")
    uf.add("c")
    uf.add("d")

    uf.union("a", "b")
    uf.union("c", "d")
    assert uf.find("a") == uf.find("b")
    assert uf.find("c") == uf.find("d")
    assert uf.find("a") != uf.find("c")

    assert uf.get("a")[1] == {"a", "b"}
    assert uf.get("b")[1] == {"a", "b"}
    assert uf.get("c")[1] == {"c", "d"}
    assert uf.get("d")[1] == {"c", "d"}

    uf.union("b", "c")
    assert uf.find("a") == uf.find("d")
    assert uf.get("a")[1] == {"a", "b", "c", "d"}

    uf.remove("b")
    assert uf.get("a")[1] == {"a", "c", "d"}
    assert uf.get("b")[1] == {"a", "c", "d"}
