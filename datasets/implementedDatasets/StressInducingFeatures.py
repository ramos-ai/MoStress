import os

import numpy as np
import pandas as pd
from datasets.Dataset import Dataset


class StressInducingFeatures(Dataset):
    def __init__(self, dataPath, dataOpts):
        self.dataPath = dataPath
        self.dataOpts = dataOpts
        self.data = self._getData()

    def _getLabel(self, labelCode):
        return self.dataOpts["stateCodes"][str(labelCode)]

    def _getData(self):

        dataList = []

        listOfSubjects = self.dataOpts["subjects"]
        listOfStressInducingSignals = self.dataOpts["signals"]
        getLabel = np.vectorize(self._getLabel)

        for subject in listOfSubjects:

            stressInducingData = {}

            subjectPath = os.path.join(
                self.dataPath, subject + ".pkl")
            subjectData = Dataset.loadData(subjectPath)
            subjectDataLabel = subjectData.get("LABEL", {}).to_numpy()
            subjectDataLength = len(subjectDataLabel)

            for stressInducingSignal in listOfStressInducingSignals:
                stressInducingData[stressInducingSignal] = subjectData.get(stressInducingSignal, {}).to_numpy()
                stressInducingData[stressInducingSignal] = stressInducingData[stressInducingSignal].reshape(
                    subjectDataLength, )

            stressInducingData["label"] = getLabel(subjectDataLabel)
            stressInducingData["label_id"] = subjectDataLabel
            stressInducingData["subject"] = subject

            dataList.append(pd.DataFrame(stressInducingData))

        return dataList
