import zipfile

with zipfile.ZipFile("1fb50640f251f77e65ca44cf269e21ee.zip", "r") as z:
    z.extractall("./")
