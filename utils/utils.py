import os


def hasFolder(folderPath):
    try:
        os.listdir(folderPath)
        return True
    except:
        return False


def createFolder(folderPath):
    if (not hasFolder(folderPath)):
        os.makedirs(folderPath)


if __name__ == "__main__":
    print(hasFolder("main/04-nbeatsFeatureExtractor/"))
