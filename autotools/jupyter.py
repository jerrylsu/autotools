"""Jupyter related tools.
"""
import os
import signal
import json
import nbformat
import subprocess as sp
import psutil
from loguru import logger


class JupyterLab(object):
    """The class of jupyterlab.
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

    def get_jupyterlab_pid(self) -> int:
        """Get jupyterlab pid.
        """
        pid_jupyterlab = -1
        for proc in psutil.process_iter(["pid", "cmdline"]):
            cmdline = proc.info["cmdline"]
            if cmdline:
                if any(["bin/jupyter-lab" in cmd for cmd in cmdline]):
                    proc_info = proc.info
                    logger.info(f"{proc_info}")
                    pid_jupyterlab = proc_info["pid"]
        return pid_jupyterlab

    def launch(self, port: int, notebook_dir: str) -> bool:
        """Launch jupyterlab server.
        """
        jupyter_server_config_json = os.path.join(os.path.expanduser('~'), ".jupyter/jupyter_server_config.json")
        jupyter_lab_config_py = os.path.join(os.path.expanduser('~'), ".jupyter/jupyter_lab_config.py")
        cmds = [
            "jupyter lab --generate-config",
            "jupyter lab password",
            f"chmod 777 {jupyter_server_config_json}",
        ]
        for cmd in cmds:
            status = self._run(cmd=cmd)
            if not status:
                return status

        # 1.Get jupyter server password.
        with open(jupyter_server_config_json, 'r') as fp:
            jupyter_server_config = json.load(fp)
            password = jupyter_server_config["ServerApp"]["password"]

        # Set config
        config_ls = [
            "c.ServerApp.terminado_settings = {'shell_command': ['/bin/bash']}\n",
            "c.NotebookApp.ip='*'\n",
            f"c.NotebookApp.password = '{password}'\n",
            "c.NotebookApp.open_browser = False\n",
            f"c.NotebookApp.port = {str(port)}\n",
            f"c.NotebookApp.notebook_dir = '{notebook_dir}'\n",
        ]
        with open(jupyter_lab_config_py, 'a+') as fp:
            fp.writelines(config_ls)

        # launch jupyterlab
        cmd = f"nohup jupyter lab --ip=0.0.0.0 --no-browser --allow-root --port {port} > ./jupyterlab.log 2>&1 &"
        status = self._run(cmd=cmd)
        if not status:
            logger.info(f"Jupyterlab server launch failed.")
            return status

        logger.info(f"Jupyterlab server launch success, port: {port}, "
                    f"pid: {self.get_jupyterlab_pid()}, "
                    f"notebook_dir: {notebook_dir}")
        return status

    def kill(self, pid: int) -> bool:
        """Kill jupyterlab server process.
        """
        status = False
        try:
            os.kill(pid, signal.SIGKILL)
            status = True
        except Exception as e:
            logger.info(str(e))
            return status
        logger.info(f"Jupyterlab server kill success, pid: {pid}")
        return status

