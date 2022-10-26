import requests
from dotenv import dotenv_values
from enum import Enum
import sys
# sys.stdout = open("./stdout.txt", "a")
# sys.stderr = open("./stderr.txt", "a")

class LogLevel(Enum):
    INFO = 1

class Logger:

    def __init__(self, component = "Logger", logLevel = LogLevel.INFO):
        self.component = component
        self.logLevel = logLevel
        self._config = dotenv_values("./.env")
        self._BOT_TOKEN = self._config.get("MOSTRESS_REPORTER_BOT_TELEGRAM_API_TOKEN")
        self._CHAT_ID = self._config.get("MOSTRESS_EXPERIMENTS_LOGS_CHAT_ID")
        self._baseUrl = f"https://api.telegram.org/bot{self._BOT_TOKEN}/sendMessage?chat_id={self._CHAT_ID}"
        self._terminal = sys.stdout
        self._stdoutFile = open("./stdout.txt", "a")
    
    def __call__(self, message):
        fullMessage = f"{self.logLevel.name}: [{self.component}] {message}"
        print(fullMessage)
        self._sendMessageToTelegramGroup(fullMessage)
    
    def _sendMessageToTelegramGroup(self, message):
        url = self._baseUrl + f"&text={message}"
        try:
            requests.get(url)
        except requests.exceptions.RequestException as e:
            msg = f"Unable to send message. Error: {e}"
            print(msg)
    
    def write(self, message):
        self._stdoutFile.write(message)
        self._terminal.write(message)
    
    def flush(self):
        pass

if __name__ == "__main__":
    logInfo = Logger("Logger Test")
    logInfo("Test Message")
