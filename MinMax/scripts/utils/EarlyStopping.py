import numpy as np
import torch
import os
import onnx
# from onnx_tf.backend import prepare
# import tensorflow as tf

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, NN_input = 0, path='checkpoint.pt', path2='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.path2 = path2
        self.trace_func = trace_func
        self.input = NN_input
    def __call__(self, val_loss, model,model2=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,model2)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,model2)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model2=None):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        model_save_directory = os.path.dirname(self.path)
        os.makedirs(model_save_directory, exist_ok=True)

        # Save PyTorch model(s)
        torch.save(model, self.path)
        if model2 is not None:
            torch.save(model2, self.path2)

        self.val_loss_min = val_loss
        
        torch.onnx.export(model,               # model being run
                          self.input,           # model input (or a tuple for multiple inputs)
                          "checkpoint.onnx",   # where to save the model (can be a file or file-like object)
                          export_params=True)
        
        
        

        
    # def export_to_h5(self, model):
    #     # Get the original .pt path
    #     pt_path = self.path  # This is path_dir in your example
    #     assert pt_path.endswith(".pt"), "Expected a .pt file"

    #     # Get the directory and base name
    #     model_save_directory = os.path.dirname(pt_path)
    #     base_name = os.path.splitext(os.path.basename(pt_path))[0]  # without .pt

    #     # Define ONNX and H5 paths
    #     onnx_path = os.path.join(model_save_directory, base_name + ".onnx")
    #     h5_path = os.path.join(model_save_directory, base_name + ".h5")

    #     # Export PyTorch model to ONNX
    #     torch.onnx.export(model, self.input, onnx_path, export_params=True)

    #     # Convert to TensorFlow
    #     onnx_model = onnx.load(onnx_path)
    #     tf_rep = prepare(onnx_model)
    #     tf_model = tf_rep.tf_module

    #     # Save as SavedModel then reload as Keras Model to write .h5
    #     saved_model_path = os.path.join(model_save_directory, base_name + "_saved_model")
    #     tf.saved_model.save(tf_model, saved_model_path)

    #     # Try loading back into a Keras Model and saving as .h5
    #     try:
    #         loaded = tf.keras.models.load_model(saved_model_path)
    #         loaded.save(h5_path)
    #         print(f"Saved .h5 model at: {h5_path}")
    #     except Exception as e:
    #         print(f"Could not convert to .h5: {e}")
    #         print(f"Model saved as SavedModel at: {saved_model_path}")




        
        
    