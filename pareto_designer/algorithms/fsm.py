from typing import TypeVar, Hashable, Generic, Iterable
from itertools import product


T_STATE = TypeVar("V", bound=Hashable)
T_CHAR = TypeVar("S", bound=Hashable)
T_TRANS = tuple[T_STATE, T_CHAR]
T_COLOR = TypeVar("C", bound=Hashable)


class FSM(Generic[T_STATE, T_CHAR]):
    """FSM `G=(V,v_init,Σ,t)`."""

    def __init__(
        self,
        V: set[T_STATE],
        Sigma: set[T_CHAR],
        t: dict[T_TRANS, T_STATE],
    ):
        """Initializes an FSM.

        Args:
            V: set of states
            Sigma: alphabet
            t: transition function: `V x Σ -> V`
        """
        self.V = V
        self.v_init = list(V)[0]
        self.Sigma = Sigma
        self._t = t

        self._pred: dict[T_STATE, set[T_TRANS]] = {}
        for v in self.V:
            self._pred[v] = set()
        for u in self.V:
            for sigma in self.Sigma:
                v = self._t[(u, sigma)]
                self._pred[v].add((u, sigma))

    def t(self, u: T_STATE, sigma: T_CHAR) -> T_STATE:
        return self._t.get((u, sigma), None)

    def pred(self, v: T_STATE) -> set[T_TRANS]:
        return self._pred.get(v, None)

    def get_outgoing_transitions(self, u: T_STATE) -> tuple[T_STATE, ...]:
        return tuple(self._t[(u, sigma)] for sigma in self.Sigma)

    def set_transition(self, u: T_STATE, sigma: T_CHAR, v: T_STATE):
        self._t[(u, sigma)] = v
        self._pred[v].add((u, sigma))

    def clean_merged_states(self, merged_states: Iterable[T_STATE], v: T_STATE):
        for w in merged_states:
            for sigma in self.Sigma:
                del self._t[(w, sigma)]
            del self._pred[w]

        for sigma in self.Sigma:
            u = self.t(v, sigma)
            self._pred[u].difference_update([(w, sigma) for w in merged_states])

        self.V.difference_update(merged_states)

    @classmethod
    def de_bruijn_fsm(cls, Sigma: set[T_CHAR], m: int):
        """Build a DB FSM (Σ^m)."""
        V: set[T_STATE] = set()
        t: dict[T_TRANS, T_STATE] = {}

        for kmer in product(Sigma, repeat=m):
            v = "".join(kmer)
            V.add(v)
            for sigma in Sigma:
                t[(v, sigma)] = f"{v[1:]}{sigma}"

        return cls[T_STATE, T_CHAR](V, Sigma, t)


class ColoredFSM(FSM, Generic[T_STATE, T_CHAR, T_COLOR]):
    """Colored FSM `G=(V,v_init,C,Σ,t,c)`."""

    def __init__(
        self,
        V: set[T_STATE],
        C: set[T_COLOR],
        Sigma: set[T_CHAR],
        t: dict[T_TRANS, T_STATE],
        c: dict[T_STATE, T_COLOR],
    ):
        """Initializes a colored FSM.

        Args:
            V: set of states
            C: set of colors
            Sigma: alphabet
            t: transition function: `V x Σ -> V`
            c: color function: `V -> C`
        """
        super(ColoredFSM, self).__init__(V, Sigma, t)
        self.recolor(C, c)

    def recolor(self, C: set[T_COLOR], c: dict[T_STATE, T_COLOR]):
        self.C = C
        self._c = c

    def c(self, v: T_STATE) -> T_COLOR:
        return self._c.get(v, None)

    def clean_merged_states(self, merged_states: Iterable[T_STATE], v: T_STATE):
        super(ColoredFSM, self).clean_merged_states(merged_states, v)
        for w in merged_states:
            del self._c[w]

    @classmethod
    def from_coloring(
        cls,
        fsm: FSM[T_STATE, T_CHAR],
        clr2patterns: dict[T_COLOR, set[T_STATE]],
    ):
        """Build a colored FSM from a given coloring."""
        C: set[T_COLOR] = set()
        c: dict[T_STATE, T_COLOR] = {}

        for clr, c_patterns in clr2patterns.items():
            for v in c_patterns:
                c[v] = clr
            C.add(clr)

        return cls[T_STATE, T_CHAR, T_COLOR](fsm.V, C, fsm.Sigma, fsm._t, c)
