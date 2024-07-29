import tensorflow as tf


# Определение кастомного коллбека для настройки вывода
class CustomCallback(tf.keras.callbacks.Callback):
    def set_params(self, params):
        super().set_params(params)
        self.params['verbose'] = 0

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.epochs = self.params['epochs']
        print(f'Starting training for {self.epochs} epochs')

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        logs = logs or {}
        loss, mae, mse = (logs.get('loss'), logs.get('mae'), logs.get('mse'))
        val_loss = logs.get('val_loss')

        if self.epochs < 10 or epoch % (self.epochs // 10) == 0 or epoch == self.epochs - 1:
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, Validation Loss: {val_loss:.4f}')
        else:
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, Validation Loss: {val_loss:.4f}', end='\r')
        
    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        logs = logs or {}
        val_loss, val_mae, val_mse = (logs.get('val_loss'), logs.get('val_mae'), logs.get('val_mse'))
        print(f'Training complete with Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, MSE: {val_mse:.4f} on Validation Data (Test Data)')