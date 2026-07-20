"""Observe immutable training events without changing the training loop."""

from D4CMPP2 import train
from D4CMPP2.src.TrainManager.callbacks import EventHistory, TrainingCallback


class ValidationPrinter(TrainingCallback):
    def on_validation_end(self, event):
        print(
            f"epoch={event.epoch} val_loss={event.val_loss} "
            f"lr={event.learning_rate}"
        )


history = EventHistory()
model_path = train(
    data="test",
    target=["Abs"],
    network="GCN",
    device="cpu",
    max_epoch=2,
    batch_size=8,
    callbacks=[ValidationPrinter(), history],
)
print(f"Saved model: {model_path}")
print("Observed events:", [event.name for event in history.events])

