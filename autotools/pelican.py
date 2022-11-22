"""Pelican related tools.
"""
import os

import pelican


class Pelican(object):
    """The class of pelican.
    """

    def __init__(self):
        pass

    def generate(self, path: str):
        """Generate the blog htmls using Pelican.
        """
        os.chdir(path)
        args = ["-s", os.path.join(path, "pelicanconf.py")]
        pelican.main(args)
