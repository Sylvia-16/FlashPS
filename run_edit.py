import json
import os
import re
import subprocess
import sys
import time

import torch
import yaml

from configs.edit_config import EditConfig


def create_launch_process(env_, edit_config_path, log_folder):

    with open(edit_config_path, "r") as f:
        config = yaml.safe_load(f)
    assert isinstance(config, dict), "Config load failed"
    edit_config = EditConfig(config)
    print("Config loaded")
    print("Edit Config", json.dumps(edit_config.__dict__, indent=4))

    launch_script = edit_config.launch_script

    log_path = f"{log_folder}/launch_script.log"

    # ! the last space is important, do not remove
    test_command = f"python3 {launch_script} --edit_config_path {edit_config_path} --log_folder {log_folder} "
    # test_command += f"--output_image_path {output_image_path} "
    test_command += f"> {log_path} 2>&1"

    print("Launching", test_command)

    p = subprocess.Popen(test_command, shell=True, env=env_)
    return [p]


def cleanup(all_procs):
    print("Cleaning up...")
    # kill all
    for p in all_procs:
        print("kill", p.pid)
        # maybe we need kill using sigkill?
        os.system(f"kill -TERM {p.pid} > /dev/null 2>&1")


def main():
    assert len(sys.argv) >= 2, "python run_edit.py <config_path>"
    project_path = os.getcwd()
    print("Project Path", project_path)

    edit_config_path = sys.argv[1]

    # create log folder
    log_storage_path = os.path.join(project_path, "logs")
    if not os.path.exists(log_storage_path):
        os.makedirs(log_storage_path)
        print("Create log folder", log_storage_path)
    else:
        print("Log folder exists", log_storage_path)

    config_name = os.path.basename(edit_config_path).split(".")[0]
    current_datetime = time.strftime("%m%d_%H%M%S")
    log_folder = os.path.join(
        log_storage_path, "{}_{}".format(current_datetime, config_name)
    )
    os.makedirs(log_folder, exist_ok=True)
    os.system(
        "cp {} {}".format(edit_config_path, log_folder)
    )  # cp config file to log folder

    env_variables = os.environ.copy()
    all_procs = []
    try:
        launch_processes = create_launch_process(
            env_variables, edit_config_path, log_folder
        )
        all_procs += launch_processes

        launch_processes[0].wait()
    except Exception as e:
        print("Error:", e)
    finally:
        cleanup(all_procs)


if __name__ == "__main__":
    main()
