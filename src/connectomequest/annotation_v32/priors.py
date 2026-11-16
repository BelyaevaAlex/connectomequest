"""Source-provided categorical priors; none of these classes can read an oracle."""
from collections.abc import Mapping, Sequence
from types import MappingProxyType
import numpy as np


class ImpossibleObservation(ValueError):
    """Observed categorical facts have zero mass under a joint model."""


def _distribution(values, label_count=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or (label_count is not None and len(values) != label_count):
        raise ValueError('probability rows must match the categorical alphabet')
    if not np.isfinite(values).all() or (values < 0).any() or values.max() <= 0:
        raise ValueError('probabilities require finite nonnegative mass')
    values = values / values.max()
    values = values / values.sum()
    values.setflags(write=False)
    return values


def _cell(value):
    if not isinstance(value, (int, np.integer)):
        raise ValueError('cell IDs must be integers, without lossy coercion')
    return int(value)


def _label(value, count):
    if not isinstance(value, (int, np.integer)) or not 0 <= value < count:
        raise ValueError('observation outside categorical alphabet')
    return int(value)


def _sample_count(count):
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('n_samples must be a positive integer')


class ProductPrior:
    """Independent cells with coupled categorical outcomes within each cell.

    Missing cells raise KeyError unless an explicit default distribution is given.
    Existing observations outside the requested cells have no effect.
    """
    def __init__(self, probabilities: Mapping[int, Sequence[float]], *, label_count=None, default=None):
        if label_count is None:
            first = next(iter(probabilities.values()), default)
            if first is None:
                raise ValueError('an alphabet or a probability row is required')
            label_count = len(first)
        if not isinstance(label_count, (int, np.integer)) or label_count < 1:
            raise ValueError('label_count must be a positive integer')
        self.label_count = int(label_count)
        self._rows = MappingProxyType({_cell(v): _distribution(p, self.label_count) for v,p in probabilities.items()})
        self._default = None if default is None else _distribution(default, self.label_count)

    def probabilities(self, cells, observed):
        result = np.empty((len(cells), self.label_count), dtype=float)
        for i,v in enumerate(cells):
            value = observed.get(v)
            if value is not None:
                result[i] = 0
                result[i, _label(value, self.label_count)] = 1
            else:
                row = self._rows.get(v, self._default)
                if row is None:
                    raise KeyError(f'no source probability row for cell {v}')
                result[i] = row
        return result

    def sample(self, cells, observed, n_samples, rng):
        _sample_count(n_samples)
        probs = self.probabilities(cells, observed)
        draws = np.empty((n_samples, len(cells)), dtype=np.int64)
        for i,row in enumerate(probs):
            draws[:,i] = rng.choice(self.label_count, size=n_samples, p=row)
        return draws


ProductCategoricalPrior = ProductPrior


class UniformProductPrior(ProductPrior):
    def __init__(self, label_count):
        super().__init__({}, label_count=label_count, default=np.ones(label_count))


class ExactJointPrior:
    """Explicit weighted categorical worlds; joint conditioning is exact.

    Observations of cells outside this declared joint domain are ignored. Such a
    model cannot claim to exploit correlations with cells outside its domain.
    None is an unresolved attempted read, not a categorical observation.
    """
    def __init__(self, cells, scenarios, weights, *, label_count=None):
        self.cells = tuple(_cell(v) for v in cells)
        if len(set(self.cells)) != len(self.cells):
            raise ValueError('joint cells must be unique')
        worlds = np.asarray(scenarios)
        if worlds.ndim != 2 or worlds.shape[1] != len(self.cells) or not len(worlds):
            raise ValueError('scenario table must have one column per cell')
        if not np.issubdtype(worlds.dtype, np.integer) or (worlds < 0).any():
            raise ValueError('scenarios must contain nonnegative categorical integers')
        if label_count is None:
            label_count = int(worlds.max()) + 1 if worlds.size else 1
        if not isinstance(label_count, (int,np.integer)) or label_count < 1 or (worlds >= label_count).any():
            raise ValueError('invalid joint alphabet')
        self.label_count = int(label_count)
        self.scenarios = worlds.astype(np.int64, copy=True)
        self.scenarios.setflags(write=False)
        self.weights = _distribution(weights)
        if len(self.weights) != len(worlds):
            raise ValueError('one weight required for each scenario')
        self._columns = {v:i for i,v in enumerate(self.cells)}

    def conditional_weights(self, observed):
        mass = self.weights.copy()
        for v,t in observed.items():
            if t is not None and v in self._columns:
                t = _label(t,self.label_count)
                mass[self.scenarios[:,self._columns[v]] != t] = 0
        total = mass.sum()
        if total <= 0:
            raise ImpossibleObservation('observed facts have zero joint probability')
        return mass / total

    def probabilities(self, cells, observed):
        weights = self.conditional_weights(observed)
        result = np.zeros((len(cells),self.label_count))
        for i,v in enumerate(cells):
            j = self._columns[v]
            result[i] = np.bincount(self.scenarios[:,j], weights=weights, minlength=self.label_count)
        return result

    def sample(self, cells, observed, n_samples, rng):
        _sample_count(n_samples)
        columns = [self._columns[v] for v in cells]
        rows = rng.choice(len(self.scenarios), size=n_samples, p=self.conditional_weights(observed))
        return self.scenarios[rows][:,columns].copy()
