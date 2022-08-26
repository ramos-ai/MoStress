from moStress.preprocessing.implementedSteps.Steps import Steps

class Normalization(Steps):

    def __init__(self, moStressPreprocessing):
        self.moStressPreprocessing = moStressPreprocessing
        self._hasNormalizationFinished = False

    def execute(self):
        try:
            if (not self._hasNormalizationFinished):
                self.moStressPreprocessing.features = [ self._rollingZScore(data) for data in self.moStressPreprocessing.features ]
                self._hasNormalizationFinished = True
        except:
            raise Exception("Normalization failed.")
            
    
    def _rollingZScore(self, df):

        copy_df = df.copy()

        for i in range(len(df) - self.moStressPreprocessing._winSize + 1):
            slicer = slice(i, self.moStressPreprocessing._winSize + i, self.moStressPreprocessing._winStep)
            mean = df[slicer].mean()
            std = df[slicer].std(ddof=0)
            copy_df[slicer] = (df[slicer] - mean) / std

        return copy_df
