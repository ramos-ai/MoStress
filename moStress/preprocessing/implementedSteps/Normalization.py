from moStress.preprocessing.implementedSteps.Steps import Steps

class Normalization(Steps):
    def __init__(self, moStressPreprocessing):
        self.moStressPreprocessing = moStressPreprocessing

    def execute(self):
        self.moStressPreprocessing.features = [ self._rollingZScore(data) for data in self.moStressPreprocessing.features ]
    
    def _rollingZScore(self, df):

        copy_df = df.copy()

        for i in range(len(df) - self.moStressPreprocessing.winSize + 1):
            slicer = slice(i, self.moStressPreprocessing.winSize + i, self.moStressPreprocessing.winStep)
            mean = df[slicer].mean()
            std = df[slicer].std(ddof=0)
            copy_df[slicer] = (df[slicer] - mean) / std

        return copy_df
