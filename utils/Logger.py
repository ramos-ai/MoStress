import requests
from dotenv import dotenv_values
from enum import Enum
import sys
from tensorflow.python import keras

class LogLevel(Enum):
    INFO = 1
    ERROR = 2

class Logger(keras.callbacks.Callback):
    def __init__(self, component="Logger", logLevel=LogLevel.INFO):
        self.component = component
        self.logLevel = logLevel
        self._config = dotenv_values("./.env")
        self._SEND_TELEGRAM_MESSAGES = self._config.get("SEND_TELEGRAM_MESSAGES")
        self._BOT_TOKEN = self._config.get("MOSTRESS_REPORTER_BOT_TELEGRAM_API_TOKEN")
        self._CHAT_ID = self._config.get("MOSTRESS_EXPERIMENTS_LOGS_CHAT_ID")
        self._baseUrl = f"https://api.telegram.org/bot{self._BOT_TOKEN}/sendMessage?chat_id={self._CHAT_ID}"
        self._outTerminal = sys.stdout
        self._errTerminal = sys.stderr

    def __call__(self, message):
        fullMessage = f"{self.logLevel.name}:[{self.component}] {message}"
        print(fullMessage)
        self._sendMessageToTelegramGroup(fullMessage)
        

    def _sendMessageToTelegramGroup(self, message):
        url = self._baseUrl + f"&text={message}"
        if (self._SEND_TELEGRAM_MESSAGES): 
            try:
                requests.get(url)
            except requests.exceptions.RequestException as e:
                msg = f"Unable to send message. Error: {e}"
                print(msg)

    def write(self, message):
        if self.logLevel == LogLevel.INFO:
            self._outTerminal.write(message)
        else:
            self._errTerminal.write(message)

    def flush(self):
        pass
    
    def on_epoch_begin(self, epoch, logs=None):
        msg = f"Epoch number {epoch} starting"
        print(msg)
        self._sendMessageToTelegramGroup(msg)

    def on_epoch_end(self, epoch, logs=None):
        msg = f"Epoch number {epoch} finished with loss: {logs['loss']}"
        print(msg)
        self._sendMessageToTelegramGroup(msg)

if __name__ == "__main__":
    logInfo = Logger("Test", LogLevel.INFO)
    logError = Logger("Test", LogLevel.ERROR)

    logInfo("Test Message 1")
    logInfo("Test Message 2")

    x = "hello"

    if not type(x) is int:
        e = TypeError("Only integers are allowed")
        logError(e)
        raise e
