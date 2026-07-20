import time
import pandas as pd
import numpy as np
import torch
from .callbacks import CallbackDispatcher, TrainingEvent
from D4CMPP2.src.utils.output import get_output

    
class Trainer():
    "The class where the operation of the training is defined"
    def __init__(self,config):
        self.config = config
        self.output = get_output(config)
        self.model_path=config['MODEL_PATH']
        self.dev=config['device']
        self.max_epoch=config['max_epoch']
        self.learning_curve=None
        self.train_error = None
        self._callback_dispatcher = CallbackDispatcher()

    def set_callbacks(self, callbacks):
        """Set runtime-only callbacks without placing objects in saved config."""

        self._callback_dispatcher = CallbackDispatcher(callbacks)

    def _event(self, name, network_manager, **kwargs):
        learning_rate = None
        if hasattr(network_manager, "get_lr"):
            learning_rate = float(network_manager.get_lr())
        checkpoint_paths = {}
        model_path = self.config.get("MODEL_PATH")
        if model_path:
            checkpoint_paths = {
                "latest": f"{model_path}/checkpoints/latest.ckpt",
                "best": f"{model_path}/checkpoints/best.ckpt",
            }
        return TrainingEvent(
            name=name,
            learning_rate=learning_rate,
            checkpoint_paths=checkpoint_paths,
            **kwargs,
        )
    
    def train(self, network_manager, train_loader, val_loader):

        self.learning_curve=getattr(network_manager,'learning_curve',None)
        start_time=time.time()
        start_epoch = getattr(network_manager, "next_epoch", 0)

        try:
            self._callback_dispatcher.emit(
                self._event("run_start", network_manager)
            )
            with self.output.progress(
                range(start_epoch, start_epoch + self.max_epoch)
            ) as t:
                for epoch in t:
                    self._callback_dispatcher.emit(
                        self._event("epoch_start", network_manager, epoch=epoch)
                    )
                    t.set_description('Epoch %d' % epoch)
                    epoch_train_loss = self.train_epoch(network_manager, train_loader)
                    epoch_train_loss=float(np.mean(epoch_train_loss))
                    self._callback_dispatcher.emit(
                        self._event(
                            "train_epoch_end",
                            network_manager,
                            epoch=epoch,
                            train_loss=epoch_train_loss,
                        )
                    )
                    epoch_val_loss = self.evaluate(network_manager, val_loader)
                    epoch_val_loss=float(np.mean(epoch_val_loss))
                    self._callback_dispatcher.emit(
                        self._event(
                            "validation_end",
                            network_manager,
                            epoch=epoch,
                            train_loss=epoch_train_loss,
                            val_loss=epoch_val_loss,
                        )
                    )
                    
                    t.set_postfix({'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss})

                    row=pd.Series({'train_loss':epoch_train_loss, 'val_loss': epoch_val_loss})
                    if self.learning_curve is None:
                        self.learning_curve=pd.DataFrame(row).transpose()
                    else:
                        self.learning_curve = pd.concat([self.learning_curve,pd.DataFrame(row).transpose()],axis=0)
                    self.learning_curve.to_csv(self.model_path+'/learning_curve.csv',index=False)
                    stopped = bool(
                        network_manager.scheduler_step(
                            epoch_val_loss,
                            completed_epoch=epoch,
                        )
                    )
                    self._callback_dispatcher.emit(
                        self._event(
                            "epoch_end",
                            network_manager,
                            epoch=epoch,
                            train_loss=epoch_train_loss,
                            val_loss=epoch_val_loss,
                            stopped=stopped,
                        )
                    )
                    if stopped:
                        break
                             
                    
        except KeyboardInterrupt as error:
            self._callback_dispatcher.emit_failure(
                self._event(
                    "interruption",
                    network_manager,
                    epoch=locals().get("epoch"),
                    error=error,
                ),
                error,
            )
            if self.config.get('legacy_silent_errors', False):
                self.output.error(
                    "[Training] Interrupted by the user; preserving legacy "
                    "silent-error behavior."
                )
            else:
                raise
        except Exception as e:
            self.train_error = e
            self._callback_dispatcher.emit_failure(
                self._event(
                    "exception",
                    network_manager,
                    epoch=locals().get("epoch"),
                    error=e,
                ),
                e,
            )
            raise
        
        network_manager.load_best_checkpoint()
        self.run_time = time.time()-start_time
        self._callback_dispatcher.emit(
            self._event(
                "run_end",
                network_manager,
                epoch=locals().get("epoch"),
            )
        )
    
    def train_epoch(self, network_manager, train_loader):
        network_manager.train()
        score_list = []
        target_list = []
        epoch_loss = 0
        count=0
        for loader in (train_loader):       
            torch.autograd.set_detect_anomaly(False)   
            y, y_pred, loss = network_manager.step(loader)
            
            score_list.append(y_pred)
            target_list.append(y)

            count+=y_pred.shape[0]
            epoch_loss += loss*y_pred.shape[0]
        if count: epoch_loss/=count

        return epoch_loss
    
    
    def evaluate(self, network_manager, val_loader):    
        network_manager.eval()
        score_list = []
        target_list = []
        epoch_loss = 0
        count=0
        with torch.no_grad():
            for loader in (val_loader):     
                torch.autograd.set_detect_anomaly(False)
                y, y_pred, loss = network_manager.step(loader)
                

                score_list.append(y_pred)
                target_list.append(y)

                count+=y_pred.shape[0]
                epoch_loss += loss*y_pred.shape[0]
        if count: epoch_loss/=count

        return epoch_loss
    
    def predict(self, network_manager, pred_loader,dropout=False):
        network_manager.eval()
        if dropout:
            network_manager.dropout_on()
        score_list = []
        target_list = []
        # OLD
        smiles_list = {}
        with torch.no_grad():
            if len(pred_loader) > 10:
                pred_loader = self.output.progress(
                    pred_loader,
                    desc="Predicting",
                    mininterval=1.0,
                )
            for loader in pred_loader:     
                torch.autograd.set_detect_anomaly(False)
                y, y_pred, loss,x = network_manager.step(loader,flag=True)
                score_list.append(y_pred)
                target_list.append(y)
                for k in x.keys():
                    if k.endswith('_smiles'):
                        _k = k[:-7]  
                        smiles_list[_k] = smiles_list.get(_k,[]) + x[k]
        return torch.cat(score_list, dim=0), torch.cat(target_list, dim=0), smiles_list

    def get_score(self, network_manager, loader):
        "Provide the score of the model. The model which is used for the training should contains the implementation of 'get_score'."
        network_manager.eval()
        pred_result = {}
        with torch.no_grad():
            if len(loader) > 10:
                loader = self.output.progress(loader, desc="Calculating scores")
            for l in loader:  
                torch.autograd.set_detect_anomaly(False)
                y_pred = network_manager.step(l,get_score=True)
                if type(y_pred) is not dict:
                    y_pred = {'prediction':y_pred}
                if pred_result:
                    for key in y_pred:
                        pred_result[key] = torch.cat([pred_result[key],y_pred[key]], dim=0)
                else:
                    pred_result = y_pred
        return pred_result
    
    
    def get_feature(self, network_manager, loader):
        "Provide the feature of the model. The model which is used for the training should contains the implementation of 'get_feature'."
        network_manager.eval()
        pred_result = {}
        with torch.no_grad():
            if len(loader) > 10:
                loader = self.output.progress(loader, desc="Calculating features")
            for l in loader:     
                torch.autograd.set_detect_anomaly(False)
                y_pred = network_manager.step(l,get_feature=True)
                if pred_result:
                    for key in y_pred:
                        pred_result[key] = torch.cat([pred_result[key],y_pred[key]], dim=0)
                else:
                    pred_result = y_pred
        return pred_result
