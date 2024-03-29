"""Git related tools.
"""
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

    def init(self) -> bool:
        """Git init.
        """
        cmd = "git init"
        status = self._run(cmd)
        return status

    def remote_add(self, url: str, name: str = "origin") -> bool:
        """Git remote add.
        """
        cmd = f"git remote add {name} {url}"
        status = self._run(cmd)
        return status

    def add(self) -> bool:
        """Git add.
        """
        cmd = "git add --all ."
        status = self._run(cmd)
        return status

    def commit(self, comment: str = "...") -> bool:
        """Git commit.
        """
        cmd = f"git commit -m {comment}"
        status = self._run(cmd)
        return status

    def push(self, branch: str = "master", force: bool = False):
        """Git push.
        """
        force = "--force" if force else ""
        cmd = f"git push origin {branch}" + force
        status = self._run(cmd)
        return status

