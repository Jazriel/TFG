from os import path, kill
import errno
import sys
from time import sleep


def pid_exists(pid):
    """Check whether pid exists in the current process table.
    UNIX only. 

    pid must be between 1 and math.inf 
    """
    try:
        kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            # ESRCH // No such process
            return False
        elif err.errno == errno.EPERM:
            # EPERM // denied access to process
            return True
        else:
            # Most likely pid is not nice 
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
        return pid_exists(pid)
    else:
        from fit_dataset import unzip_and_retrain
        pid, stdout, stderr = unzip_and_retrain(userid, datasetname, base_path=base_path)
        with open(path.join(base_path, 'images', userid, datasetname, 'pid'), mode='w') as pid_file:
            pid_file.write((str(pid)))
        if print_:
            while pid_exists(pid):
                sleep(60)
                if stdout is None:
                    break
                print('stdout: ' + str(stdout))
                print('stderr: ' + str(stderr))
        return True


if __name__ == '__main__':
    if len(sys.argv) == 3:
        print(str(is_running_or_start(userid=sys.argv[1], datasetname=sys.argv[2])).encode())
