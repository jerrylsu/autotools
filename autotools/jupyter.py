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


class NoteBook(object):
    """The class of notebook.
    """

    def __init__(self):
        pass

    def convert_notebook_to_markdown_file(self, ipynb_file_path: str, force: bool = False):
        """Convert jupyter notebook to markdown.
        """
        file_dir, ipynb_file_name = os.path.split(ipynb_file_path)
        md_file_name = ipynb_file_name.replace(".ipynb", ".md")
        md_file_path = os.path.join(file_dir, md_file_name)
        if not force and os.path.isfile(md_file_path):  # if not force and markdown file is exist.
            return
        self.format_notebook(ipynb_file_path)
        os.chdir(file_dir)
        cmd = f"jupyter nbconvert --to markdown {ipynb_file_path}"
        try:
            res = sp.run(cmd, shell=True, check=True)
            logger.info(res)
        except sp.CalledProcessError as cp:
            logger.error(f"cmd exec error: {cp}")

    def batch_convert_notebook_to_markdown_file(self, cand_dir: str, force: bool = False):
        """Convert jupyter notebook to markdown.
        """
        for root, dirs, files in os.walk(cand_dir):
            if ".ipynb_checkpoints" in root:
                continue
            for file_ in files:
                if not file_.endswith("ipynb"):
                    continue
                ipynb_file_path = os.path.join(root, file_)
                self.convert_notebook_to_markdown_file(ipynb_file_path, force=force)
                pass

    def _format_cell(self, cell: dict, cell_type: str) -> bool:
        """Format a cell in a Jupyter notebook.
        """
        if cell["cell_type"] != cell_type:
            return False
        code = cell["source"]
        lines = code.split("\n")
        if not lines:
            return False
        # remove "- " for pelican page with markdown.
        formatted = [line[2:] if line.startswith("- ") else line for line in lines]
        formatted = formatted + ["summary: Reason is the light and the light of life.", "toc: show"]
        formatted = "\n".join(formatted)
        # remove the trailing new line
        formatted = formatted.rstrip("\n")
        if formatted != code:
            cell["source"] = formatted
            return True
        return False

    def format_notebook(self, ipynb_file_path: str):
        """https://nbformat.readthedocs.io/en/latest/api.html
        """
        notebook = nbformat.read(ipynb_file_path, as_version=nbformat.NO_CONVERT)
        cell = notebook.cells[0]
        changed = self._format_cell(cell, "markdown")
        if changed:
            nbformat.write(notebook, ipynb_file_path, version=nbformat.NO_CONVERT)
            logger.info(f"The notebook {ipynb_file_path} is formatted.\n")
        else:
            logger.info(f"No change is made to the notebook {ipynb_file_path}.\n")
