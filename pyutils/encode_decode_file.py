def file_to_base64(fpath):
    """
    Get file's binary data & encode to base64 string
    """
    with open(fpath, "rb") as fin:
        return base64.b64encode(fin.read()).decode()


def base64_to_file(data, fpath):
    """
    Decode base64 input data and write back to file
    """
    with open(fpath, "wb") as fout:
        fout.write(base64.b64decode(data))
