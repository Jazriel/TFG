from subprocess import Popen
from os import mkdir, path
from zipfile import ZipFile


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
    try_mkdir(path.join(base_path, 'bottlenecks', userid, datasetname))


def retrain_and_delete(userid, datasetname, base_path):
    """
    Try to retrain inception with a different dataset, deletes the dataset after.
    Returns the pid of the process that retrains and deletes.
    """
    bottleneck_dir = path.join(base_path, 'dataset', userid, datasetname)
    output_graph_dir = path.join(base_path, 'images', userid, datasetname, 'out_graph.pb')
    output_labels_dir = path.join(base_path, 'images', userid, datasetname, 'out_labels.txt')
    image_dir = path.join(base_path, 'dataset', userid, datasetname)
    model_dir = path.join(base_path, 'graphs', 'inceptionv3_model.pb')

    try:
        retrain_process = Popen(
            'python3 tensorflow/examples/image_retraining/retrain.py '
                '--bottleneck_dir=' + bottleneck_dir + ' '
                '--how_many_training_steps 4000 '
                '--model_dir=/' + model_dir + ' '
                '--output_graph=' + output_graph_dir + ' '
                '--output_labels=' + output_labels_dir + ' '
                '--image_dir ' + image_dir + ' '
                '&& rm -rf ' + image_dir + ' ',
            shell=True)

        return retrain_process.pid, retrain_process.stdout, retrain_process
    except:

        return Popen('rm -rf ' + image_dir, shell=True).pid


def unzip(userid, datasetname, base_path):
    """Unzips the dataset."""
    zip_path = path.join(base_path, 'images', userid, datasetname+'.zip')
    extract_path = path.join(base_path, 'dataset', userid)

    with ZipFile(zip_path) as dataset:
        dataset.extractall(path=extract_path)


def unzip_and_retrain(userid, datasetname, base_path='/'):
    """Make dirs, unzip the dataset, retrain the model and delete."""
    try_mkdirs_for_training(userid, datasetname, base_path)

    unzip(userid, datasetname, base_path)

    return retrain_and_delete(userid, datasetname, base_path)
