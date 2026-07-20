import torch
import importlib.util
import os, sys
import yaml
import pandas as pd
import numpy as np
from D4CMPP2.src.utils.config_validation import validate_runtime_environment
from D4CMPP2.src.utils.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    atomic_torch_save,
    capture_rng_state,
    load_checkpoint,
    resolve_resume_checkpoint,
    restore_rng_state,
)
from D4CMPP2.src.utils.transfer_learning import (
    file_sha256,
    select_compatible_state,
    write_transfer_report,
)
from D4CMPP2.src.utils.output import get_output

class NetworkManager:

    def __init__(self, config, tf_path=None, unwrapper=None, temp=False, resume_path=None):
        
        self.config = config
        self.output = get_output(config)
        self.device = config.get('device', 'cpu')
        validate_runtime_environment(config, backend="pyg", torch_module=torch)
        self.output.info(f"[Runtime] Compute device: {self.device!r}.")
        self.last_lr = config.get('learning_rate',0.001)
        self.unwrapper = unwrapper
        self.best_loss = float('inf')
        self.es_patience = config.get('early_stopping_patience',50)
        self.es_counter = 0
        self.state= "train"
        self.tf_path = tf_path
        self.schedulers = []
        self.temp = temp
        self.resume_path = resume_path
        self.completed_epoch = -1
        self.next_epoch = 0
        self.best_epoch = None
        self.run_id = config.get("run_id")

        if self.tf_path:
            self.transferlearn_network()
        else:
            self.init_network(load_existing=not bool(self.resume_path))

        self.init_optimizer(config.get('lr_dict',{}) )
        self.init_scheduler()
        if self.resume_path:
            self.load_training_checkpoint(self.resume_path)

    def set_unwrapper(self, unwrapper):
        self.unwrapper = unwrapper

    def _output(self):
        """Return the configured adapter, including for legacy test fixtures."""

        return getattr(self, "output", get_output(getattr(self, "config", None)))

    def get_net_module(self,model_path=None):
        if model_path is None:
            model_path = self.config['MODEL_PATH']
        if os.path.exists(os.path.join(model_path,'network.py')):
            spec1 = importlib.util.spec_from_file_location("network", os.path.join(model_path,"network.py"))
            module = importlib.util.module_from_spec(spec1)
            spec1.loader.exec_module(module)
            net = getattr(module, 'network')
            return net
        network_name = self.config.get("network_id")
        if network_name is not None:
            try:
                from D4CMPP2.networks.registry import get_model

                return get_model(network_name).network
            except ValueError:
                pass
        raise FileNotFoundError(
            model_path + "/network.py not found and network_id is not registered"
        )

    # Initialize the network
    def init_network(self, load_existing=True):
        module = self.get_net_module()
        self.network = module(self.config)
        self.network.to(device = self.device)
        parameter_count = sum(
            p.numel() for p in self.network.parameters() if p.requires_grad
        )
        self._output().info(
            f"[Model] Initialized {type(self.network).__name__} with "
            f"{parameter_count:,} trainable parameters."
        )
        from D4CMPP2.networks.base import MolecularNetwork

        if isinstance(self.network, MolecularNetwork):
            self.loss_fn = self.network.compute_loss
        else:
            self.loss_fn = self.network.loss_fn

        if os.path.exists(os.path.join(self.config['MODEL_PATH'],'result','learning_curve.csv')):
            self.learning_curve = pd.read_csv(os.path.join(self.config['MODEL_PATH'],'result','learning_curve.csv'))
        elif os.path.exists(os.path.join(self.config['MODEL_PATH'],'learning_curve.csv')):
            self.learning_curve = pd.read_csv(os.path.join(self.config['MODEL_PATH'],'learning_curve.csv'))
        else:
            self.learning_curve = None

        if not load_existing:
            return
        if os.path.exists(os.path.join(self.config['MODEL_PATH'],'final.pth')):
            self.load_params(os.path.join(self.config['MODEL_PATH'],'final.pth'))
        else:
            losses = []
            for file in os.listdir(self.config['MODEL_PATH']):
                if file.startswith("param_") and file.endswith(".pth"):
                    losses.append(float(file.split("_")[1].replace(".pth","")))
            if losses:
                self.best_loss = min(losses)
                self.load_params(os.path.join(self.config['MODEL_PATH'],'param_'+str(self.best_loss)+'.pth')
                )
        

    # Transfer learning
    def transferlearn_network(self):
        self.init_network()        
        self.load_params_transfer_learn(self.tf_path)

    # Initialize the optimizer
    def init_optimizer(self,lr_dict=None):
        lr_dict = {} if lr_dict is None else lr_dict
        if not isinstance(lr_dict, dict):
            raise TypeError(
                f"lr_dict must be a mapping of layer-name components to learning rates, "
                f"got {type(lr_dict).__name__}. Example: lr_dict={{'GCNs': 0.0001}}."
            )
        invalid = {
            key: value for key, value in lr_dict.items()
            if (
                not isinstance(key, str)
                or not key.strip()
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            )
        }
        if invalid:
            raise ValueError(
                f"lr_dict keys must be non-empty layer-name strings and values must be "
                f"positive learning rates; invalid entries: {invalid!r}."
            )
        params = []
        matched_lr_keys = set()
        for name, param in self.network.named_parameters():
            matching_keys = [key for key in lr_dict if key in name.split(".")]
            if len(matching_keys) > 1:
                raise ValueError(
                    f"lr_dict entries {matching_keys!r} both match parameter {name!r}. "
                    "Use one unambiguous layer-name component for this parameter."
                )
            if matching_keys:
                key = matching_keys[0]
                matched_lr_keys.add(key)
                lr = lr_dict[key]
                self._output().info(
                    f"[Optimizer] Parameter {name!r} uses learning rate {lr:g}."
                )
            else:
                lr = self.config.get('learning_rate',0.001)
            
            params.append({'params': param, 'lr': lr})

        unmatched = sorted(set(lr_dict) - matched_lr_keys)
        if unmatched:
            available = sorted(
                {component for name, _ in self.network.named_parameters() for component in name.split(".")}
            )
            raise ValueError(
                f"lr_dict layer names {unmatched!r} did not match any trainable parameter. "
                f"Available name components include: {available!r}."
            )

        self.optimizer = getattr(torch.optim, self.config['optimizer'])(params,  
                                                                        lr=self.config.get('learning_rate',0.001), 
                                                                        weight_decay=self.config.get('weight_decay',0.0005)
                                                                        )

    # Initialize the scheduler
    def init_scheduler(self):
        self.schedulers.append( torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                    patience=self.config.get('lr_patience',10), 
                                                                    min_lr=self.config.get('min_lr',1e-7),
                                                                    factor=self.config.get('lr_plateau_decay',0.1),
                                                                    ))
        self.schedulers.append( torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                                    step_size=self.config.get('lr_step',40), 
                                                                    gamma=self.config.get('lr_step_decay',0.98)))
    
    "----------------------------------------------------------------------------------------------------------------------"
    "Below are the functions to manage the network during the training, and called by the Trainer class"

    def train(self):
        self.state = "train"
        self.network.train()

    def eval(self):
        self.state = "eval"
        self.network.eval()

    def get_lr(self):
        lrs = []
        for param_group in self.optimizer.param_groups:
            lrs.append(param_group['lr'])
        return np.mean(lrs)
        
    # One step of the training including forward and backward
    def step(self, loader, flag= False, **kargs):
        self.optimizer.zero_grad()
        if type(loader) is dict:
            loader.update({'device': self.device})
            x = self.unwrapper(**loader)
        else:
            x= self.unwrapper(*loader,device=self.device)
        y= x['target']
        x.update(kargs)
        self._validate_network_input(x)
        y_pred = self.network(**x)
        if kargs.get('get_score',False) or kargs.get('get_feature',False):
            return y_pred
                
        loss = self.loss_fn(y_pred, y)
        if self.state == "train":
            loss.backward()
            self.optimizer.step()
        if flag:
            return y, y_pred.detach(), loss.detach().item(), x
        return y, y_pred.detach(), loss.detach().item()
    
    #  One step of the prediction including forward
    def predict(self,loader):
        self.optimizer.zero_grad()
        if type(loader) is dict:
            x = self.unwrapper(**loader, device=self.device)
        else:
            x= self.unwrapper(*loader,device=self.device)
        y= x['target']
        self._validate_network_input(x)
        y_pred = self.network(**x)
        return y_pred, y

    def _validate_network_input(self, batch):
        from D4CMPP2.networks.base import MolecularNetwork

        if isinstance(self.network, MolecularNetwork):
            self.network.validate_input(batch)

    def scheduler_step(self, val_loss=None, completed_epoch=None):
        if completed_epoch is None:
            completed_epoch = getattr(self, "completed_epoch", -1) + 1
        for scheduler in self.schedulers:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # save the best model and reset the early stopping counter
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = completed_epoch
            self.save_checkpoint(val_loss)
            self.es_counter = 0
            self.save_training_checkpoint("best", completed_epoch)
        else:
            self.es_counter += 1

        # early stopping
        if self.es_counter > self.es_patience:
            self.load_best_checkpoint()
            self.completed_epoch = completed_epoch
            self.next_epoch = completed_epoch + 1
            self.save_training_checkpoint("latest", completed_epoch)
            return True

        # load the best model if the learning rate is reduced
        current_lr = self.get_lr()
        if current_lr<self.last_lr/2:
            self.last_lr=current_lr
            self.load_best_checkpoint()
            self._output().info(
                f"[Training] Learning rate reduced to {self.last_lr:g}; "
                "restored the best validation checkpoint."
            )
        self.completed_epoch = completed_epoch
        self.next_epoch = completed_epoch + 1
        self.save_training_checkpoint("latest", completed_epoch)
            
    def save_checkpoint(self, val_loss):
        self.save_params(os.path.join(self.config['MODEL_PATH'],"param_"+str(val_loss)+".pth"))

    def load_best_checkpoint(self):
        path = os.path.join(self.config['MODEL_PATH'],"param_"+str(self.best_loss)+".pth")
        if os.path.exists(path):
            self.load_params(path)
            return path
        else:
            return None

    def load_params(self, path):
        self._output().info(f"[Checkpoint] Loading model weights from {path!r}.")
        self.network.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))

    def load_params_transfer_learn(self, tf_path):
        self._output().info(
            f"[Transfer] Loading compatible parameters from {tf_path!r}."
        )
        config_path = os.path.join(tf_path, "config.yaml")
        weights_path = os.path.join(tf_path, "final.pth")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Transfer source config {config_path!r} was not found. "
                "Select a completed saved-model directory containing config.yaml."
            )
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Transfer source weights {weights_path!r} were not found. "
                "Select a completed saved-model directory containing final.pth."
            )
        try:
            with open(config_path, "r", encoding="utf-8") as stream:
                source_config = yaml.safe_load(stream)
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(
                f"Transfer source config {config_path!r} could not be read as YAML: {exc}."
            ) from exc
        if not isinstance(source_config, dict):
            raise TypeError(
                f"Transfer source config {config_path!r} must contain a YAML mapping, "
                f"got {type(source_config).__name__}."
            )

        try:
            module = self.get_net_module(tf_path)
            pretrained_network = module(source_config)
        except Exception as exc:
            raise RuntimeError(
                f"Transfer source network snapshot in {tf_path!r} could not be constructed "
                f"from config.yaml: {exc}"
            ) from exc
        try:
            params = torch.load(
                weights_path,
                weights_only=True,
                map_location=self.device,
            )
            pretrained_network.load_state_dict(params, strict=True)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Transfer source weights {weights_path!r} are incompatible with its "
                f"saved network.py/config.yaml snapshot: {exc}"
            ) from exc

        selected, report = select_compatible_state(
            pretrained_network.state_dict(),
            self.network.state_dict(),
        )
        if not selected:
            raise ValueError(
                f"Transfer source {tf_path!r} has no name-and-shape-compatible state "
                "entries for the target network. Check the selected source and target architectures."
            )
        result = self.network.load_state_dict(selected, strict=False)
        report.update(
            {
                "source_model_path": os.path.abspath(tf_path),
                "source_weights": os.path.abspath(weights_path),
                "source_weights_sha256": file_sha256(weights_path),
                "source_network": source_config.get(
                    "network_id", source_config.get("network")
                ),
                "target_network": self.config.get(
                    "network_id", self.config.get("network")
                ),
                "load_state_missing_keys": list(result.missing_keys),
                "load_state_unexpected_keys": list(result.unexpected_keys),
            }
        )
        report["report_path"] = write_transfer_report(
            report, self.config["MODEL_PATH"]
        )
        self.transfer_report = report
        counts = report["counts"]
        self._output().info(
            "[Transfer] Parameter selection complete: "
            f"loaded={counts['loaded']}, "
            f"shape_mismatch={counts['shape_mismatch']}, "
            f"source_only={counts['source_only']}, "
            f"target_only={counts['target_only']}. "
            f"Report: {report['report_path']!r}."
        )

    def save_params(self, path):
        torch.save(self.network.state_dict(), path)

    def _checkpoint_identity(self):
        return {
            "network_id": self.config.get("network_id", self.config.get("network")),
            "network_module": self.config.get("network"),
            "target": list(self.config.get("target", [])),
            "target_dim": self.config.get("target_dim"),
            "optimizer": self.config.get("optimizer"),
            "scheduler_policy": self.config.get("scheduler_policy", "legacy_dual"),
            "graph_backend": "pyg",
            "graph_schema_version": 1,
            "random_seed": self.config.get("random_seed"),
            "deterministic_algorithms": self.config.get("deterministic_algorithms", False),
            "split_strategy": self.config.get("split_strategy", "auto"),
            "scaffold_column": self.config.get("scaffold_column"),
            "scaffold_include_chirality": self.config.get(
                "scaffold_include_chirality", False
            ),
        }

    def save_training_checkpoint(self, kind, completed_epoch):
        if completed_epoch is None:
            return None
        path = os.path.join(self.config["MODEL_PATH"], "checkpoints", f"{kind}.ckpt")
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": kind,
            "run_id": getattr(self, "run_id", None),
            "completed_epoch": int(completed_epoch),
            "next_epoch": int(completed_epoch) + 1,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "schedulers": [
                {"class": scheduler.__class__.__name__, "state_dict": scheduler.state_dict()}
                for scheduler in self.schedulers
            ],
            "best_epoch": getattr(self, "best_epoch", None),
            "best_metric": self.best_loss,
            "metric_name": "val_loss",
            "early_stopping_counter": self.es_counter,
            "last_lr": self.last_lr,
            "rng_state": capture_rng_state(),
            "identity": self._checkpoint_identity(),
        }
        atomic_torch_save(payload, path)
        return path

    def load_training_checkpoint(self, value):
        path = resolve_resume_checkpoint(value)
        payload = load_checkpoint(path, self.device)
        expected = self._checkpoint_identity()
        actual = payload.get("identity", {})
        for key in (
            "network_module",
            "target",
            "target_dim",
            "optimizer",
            "scheduler_policy",
            "graph_backend",
            "graph_schema_version",
        ):
            if actual.get(key) != expected.get(key):
                raise ValueError(
                    f"Resume checkpoint {str(path)!r} is incompatible for {key}: "
                    f"checkpoint={actual.get(key)!r}, current={expected.get(key)!r}. "
                    "Use LOAD_PATH for weight-only continuation or TRANSFER_PATH for a different model/target."
                )
        for key in ("random_seed", "deterministic_algorithms"):
            if key in actual and actual.get(key) != expected.get(key):
                raise ValueError(
                    f"Resume checkpoint {str(path)!r} is incompatible for {key}: "
                    f"checkpoint={actual.get(key)!r}, current={expected.get(key)!r}. "
                    "Use the saved reproducibility policy for full resume."
                )
        for key in (
            "split_strategy",
            "scaffold_column",
            "scaffold_include_chirality",
        ):
            if key in actual and actual.get(key) != expected.get(key):
                raise ValueError(
                    f"Resume checkpoint {str(path)!r} is incompatible for {key}: "
                    f"checkpoint={actual.get(key)!r}, current={expected.get(key)!r}. "
                    "Use the saved split policy for full resume, or start a new run."
                )
        scheduler_states = payload.get("schedulers", [])
        current_names = [scheduler.__class__.__name__ for scheduler in self.schedulers]
        saved_names = [item.get("class") for item in scheduler_states]
        if saved_names != current_names:
            raise ValueError(
                f"Resume checkpoint {str(path)!r} scheduler classes {saved_names!r} "
                f"do not match current classes {current_names!r}."
            )
        self.network.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        for scheduler, item in zip(self.schedulers, scheduler_states):
            scheduler.load_state_dict(item["state_dict"])
        self.best_epoch = payload.get("best_epoch")
        self.best_loss = payload["best_metric"]
        self.es_counter = payload["early_stopping_counter"]
        self.last_lr = payload["last_lr"]
        self.completed_epoch = payload["completed_epoch"]
        self.next_epoch = payload["next_epoch"]
        self.parent_run_id = payload.get("run_id")
        restore_rng_state(payload["rng_state"])
        self._output().info(
            f"[Checkpoint] Restored full training state from {str(path)!r}; "
            f"next epoch: {self.next_epoch}."
        )
        return str(path)

    def dropout_on(self):
        for m in self.network.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
