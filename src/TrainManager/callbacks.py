"""Small, observation-only callback contract for the training loop."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple


TRAINING_EVENTS: Tuple[str, ...] = (
    "run_start",
    "epoch_start",
    "train_epoch_end",
    "validation_end",
    "epoch_end",
    "run_end",
    "exception",
    "interruption",
)


@dataclass(frozen=True)
class TrainingEvent:
    """Immutable snapshot exposed to callbacks.

    Model, optimizer, dataloader, and mutable config objects are intentionally
    excluded so callbacks do not become coupled to manager internals.
    """

    name: str
    epoch: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    stopped: bool = False
    checkpoint_paths: Mapping[str, str] = field(default_factory=dict)
    error: Optional[BaseException] = None

    def __post_init__(self) -> None:
        if self.name not in TRAINING_EVENTS:
            raise ValueError(
                f"Unknown training event {self.name!r}. "
                f"Supported events: {list(TRAINING_EVENTS)!r}."
            )
        object.__setattr__(
            self,
            "checkpoint_paths",
            MappingProxyType(dict(self.checkpoint_paths)),
        )


class TrainingCallback:
    """Optional callback base class.

    Subclasses may implement ``on_event(event)`` for every event or a specific
    method such as ``on_validation_end(event)``. Specific handlers run first.
    """

    def on_event(self, event: TrainingEvent) -> None:
        """Observe a training event."""


class EventHistory(TrainingCallback):
    """Built-in observer that keeps immutable event snapshots in memory."""

    def __init__(self) -> None:
        self.events: list[TrainingEvent] = []

    def on_event(self, event: TrainingEvent) -> None:
        self.events.append(event)


class CallbackDispatcher:
    """Validate callbacks and dispatch events in registration order."""

    def __init__(self, callbacks: Optional[Sequence[Any]] = None) -> None:
        if callbacks is None:
            callbacks = ()
        if isinstance(callbacks, (str, bytes)) or not isinstance(callbacks, Sequence):
            raise TypeError(
                "callbacks must be a sequence of callback objects, for example "
                "[MyCallback()]."
            )
        self.callbacks = tuple(callbacks)
        for index, callback in enumerate(self.callbacks):
            if callback is None:
                raise TypeError(f"callbacks[{index}] is None; provide a callback object.")
            has_generic = callable(getattr(callback, "on_event", None))
            has_specific = any(
                callable(getattr(callback, f"on_{name}", None))
                for name in TRAINING_EVENTS
            )
            if not has_generic and not has_specific:
                raise TypeError(
                    f"callbacks[{index}] ({type(callback).__name__}) does not define "
                    "on_event(event) or a supported on_<event>(event) method."
                )

    def emit(self, event: TrainingEvent) -> None:
        """Dispatch one immutable event, propagating callback failures."""

        for callback in self.callbacks:
            specific = getattr(callback, f"on_{event.name}", None)
            if callable(specific):
                specific(event)
            generic = getattr(callback, "on_event", None)
            if callable(generic):
                generic(event)

    def emit_failure(self, event: TrainingEvent, original: BaseException) -> None:
        """Notify failure handlers without hiding the original exception."""

        try:
            self.emit(event)
        except BaseException as callback_error:
            add_note = getattr(original, "add_note", None)
            if callable(add_note):
                add_note(
                    "A callback failure handler also failed: "
                    f"{type(callback_error).__name__}: {callback_error}"
                )
