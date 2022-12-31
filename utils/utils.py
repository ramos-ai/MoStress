import os
import gc


def hasFolder(folderPath):
    try:
        os.listdir(folderPath)
        return True
    except:
        return False


def hasDataFile(dataFilePath):
    return os.path.isfile(dataFilePath)


def createFolder(folderPath):
    if (not hasFolder(folderPath)):
        os.makedirs(folderPath)

def deleteData(data):
    del(data)
    gc.collect()


if __name__ == "__main__":
    print(hasFolder("main/04-nbeatsFeatureExtractor/"))
