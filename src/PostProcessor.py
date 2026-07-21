import os
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import pickle
import re
import uuid

from .utils.metrics import get_metrics
from .utils import tools
from .utils.output import get_output

import traceback
class PostProcessor:
    """
    The class for the postprocessing of the trained model.
    This class is used to save the model, learning curve, prediction, and metrics after the training.
    """

    def __init__(self, config):
        self.config = config
        self.output = get_output(config)
        self.result_path = config['MODEL_PATH']+"/result"
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    def postprocess(self, dm, nm, tm, train_loaders, val_loaders, test_loaders):
        "The main function for the postprocessing"
        try:
            self.save_final_model(nm)
            self.save_scaler(dm.scaler)
        except Exception:
            if self.config.get('legacy_silent_errors', False):
                self.output.error(
                    "[Results] Model saving failed while writing the final model."
                )
                self.output.error(traceback.format_exc())
                return
            raise
        
        try:
            tm.learning_curve.to_csv(os.path.join(self.result_path,"learning_curve.csv"),index=False)
            self.draw_learning_curve(tm.learning_curve)
        except Exception:
            if self.config.get('legacy_silent_errors', False):
                self.output.error("[Results] Failed to save the learning curve.")
                self.output.error(traceback.format_exc())
            else:
                raise
            
        self.save_model_summary(nm.network,tm.run_time)

        try:
            if self.config.get("save_prediction",True):
                prediction_df = self.save_prediction(nm, tm, train_loaders, val_loaders, test_loaders, scaler=dm.scaler)
                self.plot_prediction(self.config.get("target"),prediction_df)
                self.save_metrics(self.config.get("target"),prediction_df)
        except Exception:
            if self.config.get('legacy_silent_errors', False):
                self.output.error("[Results] Failed to save predictions or metrics.")
                self.output.error(traceback.format_exc())
            else:
                raise
        
    def save_prediction(self, nm, tm, train_loader, val_loader, test_loader,scaler=None):
        train_pred, train_true, train_smiles = tm.predict(nm, train_loader)
        val_pred, val_true, val_smiles = tm.predict(nm, val_loader)
        test_pred, test_true, test_smiles = tm.predict(nm, test_loader)
        targets = self.config.get("target")
        if scaler is not None:
            train_true = scaler.inverse_transform(train_true.cpu().numpy())
            val_true = scaler.inverse_transform(val_true.cpu().numpy())
            test_true = scaler.inverse_transform(test_true.cpu().numpy())
            train_pred = scaler.inverse_transform(train_pred.cpu().numpy())
            val_pred = scaler.inverse_transform(val_pred.cpu().numpy())
            test_pred = scaler.inverse_transform(test_pred.cpu().numpy())

        prediction_df = None
        for true,pred,smiles,sets in zip([train_true,val_true,test_true],[train_pred,val_pred,test_pred],[train_smiles,val_smiles,test_smiles],['train','val','test']):
            data = {}
            for i, target in enumerate(targets):
                data[f"{target}_pred"] = pred[:, i]
                data[f"{target}_true"] = true[:, i]

            data.update({k: smiles[k] for k in smiles.keys()})
            data['set'] = sets

            df = pd.DataFrame(data)
            if prediction_df is None:
                prediction_df = df
            else:
                prediction_df = pd.concat([prediction_df,df],axis=0)

        prediction_df.to_csv(os.path.join(self.result_path,"prediction.csv"),index=False)
        return prediction_df

    def save_metrics(self, targets, prediction_df):
        metrics = get_metrics(targets, prediction_df)
        for set in ['train','val','test']:
            table = PrettyTable()
            col = ['target',f'{set}_r2',f'{set}_mae',f'{set}_rmse']
            table.field_names = col
            for t in targets:
                table.add_row([t,metrics[f'{set}_r2'][t],metrics[f'{set}_mae'][t],metrics[f'{set}_rmse'][t]])
            self.output.info(f"[Results] Metrics for split {set!r}:\n{table}")
        
        metrics.to_csv(os.path.join(self.result_path,"metrics.csv"))

    def plot_prediction(self,target, prediction_df):
        img = tools.plot_prediction(target, prediction_df)
        img.save(os.path.join(self.result_path,"prediction.png"))
        

    def save_model_summary(self, model,run_time):
        with open(os.path.join(self.config['MODEL_PATH'],"model_summary.txt"), "w") as file:
            file.write(str(model))
            file.write(f"#params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            file.write("\n")
            file.write(str(model.__class__.__name__))
            file.write("\n")
            file.write(str(model.__class__.__doc__))
            file.write("\n")
            file.write("learning_time: "+str(run_time))

    def draw_learning_curve(self, learning_curve):
        img = tools.plot_learning_curve(learning_curve['train_loss'],learning_curve['val_loss'])
        img.save(os.path.join(self.result_path,"learning_curve.png"))

    def clear_checkpoint(self):
        checkpoint_pattern = re.compile(
            r"^param_[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\.pth$"
        )
        for file in os.listdir(self.config['MODEL_PATH']):
            if checkpoint_pattern.fullmatch(file):
                os.remove(os.path.join(self.config['MODEL_PATH'], file))

    def save_scaler(self, scaler):
        """Atomically save the fitted target scaler required by Analyzer."""

        if scaler is None:
            return
        model_path = self.config['MODEL_PATH']
        scaler_path = os.path.join(model_path, "scaler.pkl")
        staging_path = os.path.join(
            model_path,
            f".scaler.pkl.{uuid.uuid4().hex}.tmp",
        )
        try:
            with open(staging_path, "wb") as file:
                pickle.dump(scaler, file)
            if not os.path.isfile(staging_path) or os.path.getsize(staging_path) == 0:
                raise OSError(
                    f"Scaler staging file {staging_path!r} was not created as a "
                    "non-empty regular file. The previous scaler.pkl was preserved."
                )
            os.replace(staging_path, scaler_path)
        finally:
            if os.path.exists(staging_path):
                try:
                    os.remove(staging_path)
                except OSError:
                    pass

    def save_final_model(self, nm):
        """Atomically commit final.pth before removing package checkpoints."""
        model_path = self.config['MODEL_PATH']
        final_path = os.path.join(model_path, "final.pth")
        staging_path = os.path.join(model_path, f".final.pth.{uuid.uuid4().hex}.tmp")
        try:
            nm.save_params(staging_path)
            if not os.path.isfile(staging_path) or os.path.getsize(staging_path) == 0:
                raise OSError(
                    f"Final model staging file {staging_path!r} was not created as a non-empty regular file. "
                    "The previous final.pth and best checkpoints were preserved."
                )
            os.replace(staging_path, final_path)
            self.clear_checkpoint()
        finally:
            if os.path.exists(staging_path):
                try:
                    os.remove(staging_path)
                except OSError:
                    pass
        
                
