#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import argparse
import os
import re
import subprocess
from typing import Dict, List, Optional, Type


DEFAULT_MANAGER = "venv"
DEFAULT_NAME = "gtc"
REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class VManager(abc.ABC):
    @classmethod
    def create(cls, py_version: str, venv_path: str) -> None:
        cmds = cls.make_create_commands(py_version, venv_path)
        print(f"Creating virtual environment at '{os.path.abspath(venv_path)}' ...")
        cls.run_cmds(cmds)

    @classmethod
    def install(cls, requirements: str, venv_path: Optional[str] = None) -> None:
        cmds = cls.make_install_commands(requirements, venv_path)
        cls.run_cmds(cmds)

    @classmethod
    @abc.abstractmethod
    def make_create_commands(cls, py_version: str, venv_path: str) -> List[str]:
        ...

    @classmethod
    @abc.abstractmethod
    def make_install_commands(cls, requirements: str, venv_path: Optional[str] = None) -> List[str]:
        ...

    @classmethod
    def run_cmds(cls, cmds: List[str], exit_code: Optional[int] = -10) -> None:
        try:
            for cmd in cmds:
                print(cmd)
                subprocess.check_call(cmd.split())
        except subprocess.CalledProcessError as e:
            print(f"ERROR!\n{e}")
            exit(exit_code)


class VenvManager(VManager):
    @classmethod
    def make_create_commands(cls, py_version: str, venv_path: str) -> List[str]:
        return [
            f"python{py_version} -m venv {venv_path}",
            f"{venv_path}/bin/pip3 install --upgrade wheel pip setuptools",
        ]

    @classmethod
    def make_install_commands(cls, requirements: str, venv_path: Optional[str] = None) -> List[str]:
        pip_prefix = f"{venv_path}/bin/pip3" if venv_path else "pip3"
        if requirements == "run":
            return [f"{pip_prefix} install -e {REPO_ROOT}[all]"]
        elif requirements == "develop":
            return [f"{pip_prefix} install -e {REPO_ROOT}[all] -r {REPO_ROOT}/requirements_dev.txt"]
        else:
            return []


class VirtualenvManager(VenvManager):
    @classmethod
    def make_create_commands(cls, py_version: str, venv_path: str) -> List[str]:
        return [f"virtualenv -p python{py_version} {venv_path}"]


class CondaManager(VManager):
    @classmethod
    def make_create_commands(cls, py_version: str, venv_path: str) -> List[str]:
        return [f"conda create -p {venv_path} -y python={py_version}"]

    @classmethod
    def make_install_commands(cls, requirements: str, venv_path: Optional[str] = None) -> List[str]:
        pip_prefix = f"conda run -p {venv_path} pip3" if venv_path else "pip3"
        if requirements == "run":
            return [f"{pip_prefix} install -e {REPO_ROOT}[all]"]
        elif requirements == "develop":
            return [f"{pip_prefix} install -e {REPO_ROOT}[all] -r {REPO_ROOT}/requirements_dev.txt"]
        else:
            return []


MANAGERS: Dict[str, Type[VManager]] = {
    "venv": VenvManager,
    "virtualenv": VirtualenvManager,
    "conda": CondaManager,
}


# -- Parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "name",
    default=DEFAULT_NAME,
    nargs="?",
    help="virtual environment name (default: '%(default)s')",
)
parser.add_argument("-p", "--python", default="3.8", help="python version (default: %(default)s)")
parser.add_argument(
    "-d", "--dir", default=".venvs", help="parent directory for the venv (default: %(default)s)"
)
parser.add_argument(
    "-m",
    "--manager",
    choices=list(MANAGERS.keys()),
    default=DEFAULT_MANAGER,
    help="virtual environment manager (default: %(default)s)",
)
parser.add_argument(
    "-r",
    "--requirements",
    choices=["none", "run", "develop"],
    default="develop",
    help="select requirements to install after creation (default: %(default)s)",
)
parser.add_argument(
    "-i",
    "--install-only",
    action="store_true",
    help="skip venv creation and install packages in current environment (default: %(default)s)",
)

args = parser.parse_args()
print("\nOptions:")
for key in sorted(vars(args)):
    print(f"    {key}: {getattr(args, key)}")
print("")

if not re.match(r"3.\d\d?(\.\d\d?)?$", args.python):
    print(f"\nError: invalid Python version specification ('{args.python}')\n")
    exit(-1)

# -- Install
manager = MANAGERS[args.manager]
venv_path = None

if not args.install_only:
    venv_path = f"{args.dir}/{args.name}"
    manager.create(args.python, venv_path)

if args.requirements != "none":
    manager.install(args.requirements, venv_path)

print("Done")
