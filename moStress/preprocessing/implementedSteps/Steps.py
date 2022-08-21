from abc import ABC, abstractmethod

class Steps(ABC):
    @abstractmethod
    def execute(self):
        pass