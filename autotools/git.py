"""Git related tools.
"""
import os
import subprocess as sp
from loguru import logger


class Git(object):
    """The class of Git.
    """

    def __init__(self, path: str):
        self.path = path
        self.

    def _run(self, cmd: str) -> bool:
        """Subprocess run.
        """
        try:
            ret = sp.run(cmd, shell=True, check=True)
            return ret.returncode == 0
        except sp.CalledProcessError as cpe:
            logger.error(f"Cmd exec error: {cpe}.")
            return False

    def init(self) -> bool:
        """Git init.
        """
        os.chdir(self.path)
        cmd = "git init"
        status = self._run(cmd)
        return status

    def remote_add(self, url: str, name: str = "origin") -> bool:
        """Git remote add.
        """
        os.chdir(self.path)
        cmd = f"git remote add {name} {url}"
        status = self._run(cmd)
        return status

    def add(self) -> bool:
        """Git add.
        """
        os.chdir(self.path)
        cmd = "git add --all ."
        status = self._run(cmd)
        return status

    def commit(self, comment: str = "...") -> bool:
        """Git commit.
        """
        os.chdir(self.path)
        cmd = f"git commit -m {comment}"
        status = self._run(cmd)
        return status

    def push(self, branch: str = "master", force: bool = False):
        """Git push.
        """
        os.chdir(self.path)
        cmd = f"git push origin {branch}" + "--force" if force else ""
        status = self._run(cmd)
        return status

