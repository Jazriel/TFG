from scripts.run_inference import run_inference_on_image
from scripts.maybe_start_retrain import is_running_or_start
from os import path, remove


def test_run_inference():
    """Runs the panda image through the inception model."""
    assert 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca' \
           in run_inference_on_image('../images/cropped_panda.jpg',
                                     '../inceptionv3_model.pb',
                                     '../inceptionv3_labels.txt',
                                     '../inceptionv3_label_map_proto.pbtxt')


def test_maybe_start_retrain():
    assert (is_running_or_start(
                userid='0', 
                datasetname='myds',
                base_path='../',
                print_=True)
            is True)
    assert path.isfile('../images/0/myds/pid')
    remove('../images/0/myds/pid')