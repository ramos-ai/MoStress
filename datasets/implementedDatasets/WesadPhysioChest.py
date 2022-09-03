import os

import numpy as np
import pandas as pd
from datasets.Dataset import Dataset


class WesadPhysioChest(Dataset):
    def __init__(self, dataPath, dataOpts):
        self.dataPath = dataPath
        self.dataOpts = dataOpts
        self.data = self._getData()

    def _getLabel(self, labelCode):
        return self.dataOpts["stateCodes"][str(labelCode)]

    def _getData(self):

        dataList = []

        listOfSubjects = self.dataOpts["subjects"]
        listOfChestPhysioSignals = self.dataOpts["signals"]
        getLabel = np.vectorize(self._getLabel)

        for subject in listOfSubjects:

            chestPhysioData = {}

            subjectPath = os.path.join(
                self.dataPath, subject, subject + ".pkl")
            subjectData = Dataset.loadData(subjectPath)
            subjectDataLabel = subjectData["label"]
            subjectDataLength = len(subjectDataLabel)

            for chestPhysioSignal in listOfChestPhysioSignals:
                chestPhysioData[chestPhysioSignal] = subjectData["signal"]["chest"][chestPhysioSignal]
                chestPhysioData[chestPhysioSignal] = chestPhysioData[chestPhysioSignal].reshape(
                    subjectDataLength, )

            self._adjustUnnecessaryLabelCode(subjectDataLabel, 7, 5)
            self._adjustUnnecessaryLabelCode(subjectDataLabel, 6, 5)

            chestPhysioData["label"] = getLabel(subjectDataLabel)
            chestPhysioData["label_id"] = subjectDataLabel
            chestPhysioData["subject"] = subjectData["subject"]

            dataList.append(pd.DataFrame(chestPhysioData))

        return dataList
