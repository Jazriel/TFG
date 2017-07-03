from os import path, kill
import sys
from time import sleep

def check_pid(pid):
    """Check if the pid exists (the process is running)"""
    try:
        kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def is_running_or_start(userid, datasetname, base_path='/', print_=False):
    """
    Check if the file is running or not. If it has not been executed start it.
    Return True if it started or is running. False if it has finished.
    """
    if path.isfile(path.join(base_path, 'images', userid, datasetname, 'pid')):
        with open(path.join(base_path, 'images', userid, datasetname, 'pid'), mode='r') as pid:
            pid = int(pid.readline().strip())
        return check_pid(pid)
    else:
        from fit_dataset import unzip_and_retrain
        pid, stdout, stderr = unzip_and_retrain(userid, datasetname, base_path=base_path)
        with open(path.join(base_path, 'images', userid, datasetname, 'pid'), mode='w') as pid_file:
            pid_file.write((str(pid)))
        if print_:
            while check_pid(pid):
                sleep(60)
                if stdout is None:
                    break
                print('stdout: ' + str(stdout))
                print('stderr: ' + str(stderr))
        return True


if __name__ == '__main__':
    if len(sys.argv) == 3:
        print(str(is_running_or_start(userid=sys.argv[1], datasetname=sys.argv[2])).encode())
