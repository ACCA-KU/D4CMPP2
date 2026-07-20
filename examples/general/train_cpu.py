"""Two-epoch CPU training smoke example."""

from D4CMPP2 import Analyzer, train


model_path = train(
    data="test",
    target=["Abs"],
    network="GCN",
    device="cpu",
    max_epoch=2,
    batch_size=4,
)
print(model_path)
analyzer = Analyzer(model_path, save_result=False)
print(analyzer.predict(["CCO", "CCN"]))
