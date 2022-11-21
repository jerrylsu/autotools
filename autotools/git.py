"""Git related tools.
"""
import os
import subprocess as sp
from loguru import logger


class Git(object):
    """The class of Git.
    """

    def __init__(self):
        pass

    def _run(self, cmd: str) -> bool:
        """Subprocess run.
        """
        try:
            ret = sp.run(cmd, shell=True, check=True)
            return ret.returncode == 0
        except sp.CalledProcessError as cpe:
            logger.error(f"Cmd exec error: {cpe}.")
            return False

    def init(self, path: str) -> bool:
        """Git init.
        """
        os.chdir(path)
        cmd = "git init"
        status = self._run(cmd)
        return status

    def remote_add(self, url: str, path: str, name: str = "origin") -> bool:
        """Git remote add.
        """
        os.chdir(path)
        cmd = f"git remote add {name} {url}"
        status = self._run(cmd)
        return status

    def add(self, path: str) -> bool:
        """Git add.
        """
        os.chdir(path)
        cmd = "git add --all ."
        status = self._run(cmd)
        return status

    def commit(self, path: str, comment: str = "...") -> bool:
        """Git commit.
        """
        os.chdir(path)
        cmd = f"git commit -m {comment}"
        status = self._run(cmd)
        return status

    def push(self, path: str, branch: str = "master", force: bool = False):
        """Git push.
        """
        os.chdir(path)
        cmd = f"git push origin {branch}" + "--force" if force else ""
        status = self._run(cmd)
        return status

