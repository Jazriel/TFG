import errno, sys
from time import sleep
from subprocess import Popen
from os import mkdir, path, kill
from zipfile import ZipFile
from glob import glob

def try_mkdir(path):
    """Try to make a directory."""
    try:
        mkdir(path=path)
    except:
        pass


def try_mkdirs_for_training(userid, datasetname, base_path):
    """Try to make the directories used for training."""
    try_mkdir(path.join(base_path, 'dataset', userid, datasetname))
    try_mkdir(path.join(base_path, 'images', userid, datasetname))


def retrain(userid, datasetname, base_path):
    """
    Try to retrain inception with a different dataset, deletes the dataset after.
    Returns the pid of the process that retrains and deletes.
    """
    output_graph_dir = path.join(base_path, 'images', userid, datasetname, 'out_graph.pb')
    output_labels_dir = path.join(base_path, 'images', userid, datasetname, 'out_labels.txt')
    image_dir, *els = glob(path.join(base_path, 'dataset', userid, datasetname, '**'))
    model_dir = path.join(base_path, 'graphs')

    try:
        retrain_process = Popen(
            'python3 tensorflow/tensorflow/examples/image_retraining/retrain.py '
            '--model_dir ' + model_dir + ' '
            '--output_graph ' + output_graph_dir + ' '
            '--output_labels ' + output_labels_dir + ' '
            '--image_dir ' + image_dir + ' ',
            shell=True)

        return retrain_process.pid, retrain_process.stdout, retrain_process
    except:
        raise


def unzip(userid, datasetname, base_path):
    """Unzips the dataset."""
    zip_path = path.join(base_path, 'images', userid, datasetname+'.zip')
    extract_path = path.join(base_path, 'dataset', userid, datasetname)

    with ZipFile(zip_path) as dataset:
        dataset.extractall(path=extract_path)


def unzip_and_retrain(userid, datasetname, base_path='/'):
    """Make dirs, unzip the dataset, retrain the model and delete."""
    try_mkdirs_for_training(userid, datasetname, base_path)
    print('Directories for retrain done')

    unzip(userid, datasetname, base_path)
    print('Unzipped dataset')

    return retrain(userid, datasetname, base_path)

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
