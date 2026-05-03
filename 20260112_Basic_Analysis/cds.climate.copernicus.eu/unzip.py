import zipfile

with zipfile.ZipFile("3353f65a26b92d69379f78c788877008.zip", "r") as z:
    z.extractall("./")
