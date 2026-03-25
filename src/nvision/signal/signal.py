"""Signal model and parameter abstractions for Bayesian localization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Parameter:
    """A parameter with a known value (ground truth or point estimate).

    Base class for parameters. Represents a parameter with a single value
    and no uncertainty. Used in TrueSignal for ground truth parameters.

    Attributes
    ----------
    name : str
        Parameter name (e.g., 'frequency', 'linewidth')
    bounds : tuple[float, float]
        (min, max) bounds for this parameter
    value : float
        Parameter value
    """

    name: str
    bounds: tuple[float, float]
    value: float

    def __post_init__(self) -> None:
        """Validate parameter is within bounds."""
        if not (self.bounds[0] <= self.value <= self.bounds[1]):
            raise ValueError(f"Parameter {self.name} value {self.value} outside bounds {self.bounds}")

    def mean(self) -> float:
        """Get parameter value (for consistency with ParameterWithPosterior)."""
        return self.value

    def uncertainty(self) -> float:
        """Get uncertainty (zero for known parameters)."""
        return 0.0

    def entropy(self) -> float:
        """Get entropy (zero for known parameters)."""
        return 0.0

    def converged(self, threshold: float) -> bool:
        """Check convergence (always True for known parameters)."""
        return True


class SignalModel(ABC):
    """Abstract signal model defining the shape of the signal.

    Stateless. Knows how to compute the signal given parameter values that match
    **this** concrete type. Shared between :class:`TrueSignal` and beliefs.

    **Substitutability:** Subclasses share the same method *names* and broad types,
    but each expects a **different** parameter set (:meth:`parameter_names`). You
    cannot drop in another ``SignalModel`` subclass in place of a specific one and
    keep the same ``params`` list—callers must always pair a model instance with a
    matching bundle of parameters. That coupling is not expressed in the type
    system (a single ``list[Parameter]`` erases the per-model schema); it is a
    deliberate tradeoff, not a claim of full Liskov substitutability on ``params``.

    **Explicit evaluation:** Each concrete model should expose ``eval_<snake_case_class_name>``
    (e.g. :meth:`~nvision.signal.gaussian.GaussianModel.eval_gaussian_model`) taking
    probe ``x`` and **physical parameters as separate floats** in a documented order
    matching :meth:`parameter_names`. Callers that know the model type should call
    that method directly. :meth:`compute` stays the polymorphic entry for beliefs,
    :class:`TrueSignal`, and :class:`~nvision.signal.composite.CompositePeakModel`.
    """

    @abstractmethod
    def compute(self, x: float, params: list[Parameter]) -> float:
        """Compute signal value at position x given parameters.

        ``params`` must include exactly the names returned by :meth:`parameter_names`
        for **this** instance. Pass them in that same order to avoid a dict lookup in
        :meth:`_param_floats_canonical` (used by hot ``compute`` implementations).

        **Contract:** Use ``params`` only while this method runs—read ``value`` /
        ``name`` / ``bounds``, do not retain ``params`` or elements and assume
        their values stay fixed. Callers (e.g. SMC, unit-cube wrappers) may reuse
        the same ``Parameter`` instances and assign ``.value`` between calls.

        Parameters
        ----------
        x : float
            Position to evaluate signal
        params : list[Parameter]
            List of parameters

        Returns
        -------
        float
            Signal value at x
        """
        pass

    def gradient(self, x: float, params: list[Parameter]) -> dict[str, float] | None:
        """Compute analytical gradient of the signal with respect to parameters.

        Optional. If implemented, allows locators to use mathematical derivations
        like Fisher Information for variance and entropy.

        Parameters
        ----------
        x : float
            Position to evaluate signal
        params : list[Parameter]
            List of parameters

        Returns
        -------
        dict[str, float] | None
            Dictionary of partial derivatives {param_name: d_signal/d_param},
            or None if gradients are not analytically available.
        """
        return None

    @abstractmethod
    def parameter_names(self) -> list[str]:
        """Return ordered list of parameter names this model expects."""
        pass

    def _params_to_dict(self, params: list[Parameter]) -> dict[str, float]:
        """Convert parameter list to dict for easier access."""
        return {p.name: p.value for p in params}

    def _param_floats_canonical(self, params: list[Parameter]) -> tuple[float, ...]:
        """Parameter values in :meth:`parameter_names` order.

        If ``params`` is already in that order (typical for unit-cube wrappers, SMC
        scratch, and SBED sample lists), no dict is allocated. Otherwise falls back
        to a name lookup dict once per call.
        """
        names = self.parameter_names()
        n = len(names)
        if len(params) != n:
            msg = f"{type(self).__name__}: expected {n} parameters, got {len(params)}"
            raise ValueError(msg)
        if n == 0:
            return ()
        if all(params[i].name == names[i] for i in range(n)):
            return tuple(float(params[i].value) for i in range(n))
        by_name = {p.name: float(p.value) for p in params}
        vals: list[float] = []
        for ni in names:
            if ni not in by_name:
                have = set(by_name)
                raise ValueError(f"{type(self).__name__}: missing parameter {ni!r}; have {have}, need {set(names)}")
            vals.append(by_name[ni])
        return tuple(vals)


@dataclass
class TrueSignal:
    """Ground truth signal with fixed parameters.

    Only exists in simulation — in real hardware this is unknown.
    Produces measurements via __call__.

    Attributes
    ----------
    model : SignalModel
        The signal model defining the shape
    parameters : list[Parameter]
        Exact parameter values, no uncertainty
    """

    model: SignalModel
    parameters: list[Parameter]

    def __post_init__(self) -> None:
        """Validate parameters match model."""
        expected = set(self.model.parameter_names())
        actual = {p.name for p in self.parameters}
        if expected != actual:
            raise ValueError(f"Parameters don't match model. Expected {expected}, got {actual}")

    def __call__(self, x: float) -> float:
        """Evaluate true signal at position x.

        Parameters
        ----------
        x : float
            Position to evaluate

        Returns
        -------
        float
            True signal value
        """
        return self.model.compute(x, self.parameters)

    def get_param(self, name: str) -> Parameter:
        """Get parameter by name."""
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"Parameter {name} not found")
