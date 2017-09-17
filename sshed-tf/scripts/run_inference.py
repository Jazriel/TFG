import numpy as np
import tensorflow as tf
import sys


def create_graph(model_path):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image_path='/images/panda.jpg',
                           model_path='/graphs/inceptionv3_model.pb',
                           labels_path='/graphs/inceptionv3_labels.txt',
                           tensor='softmax:0',
                           model_to_labels_path=None):
    answer = None
    for file in (image_path, model_path, labels_path):
        if not tf.gfile.Exists(file):
            tf.logging.fatal('File does not exist %s', file)
            return answer

    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Creates graph from saved GraphDef.
    # Don't need to save the graph tf does it behind the scenes.

    create_graph(model_path)

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name(tensor)
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-3:][::-1]  # Getting top 3 predictions

        labels = dict()
        labels_lines = None
        with open(labels_path, 'rb') as f:
            labels_lines = f.readlines()

        if model_to_labels_path is not None:

            parse_labels_to_readable_labels(labels, labels_lines, model_to_labels_path)

        else:
            for i in range(len(labels_lines)):
                labels[i] = labels_lines[i].decode()
        try:
            for i in range(3):
                answer = labels[top_k[i]]
                score = predictions[top_k[i]]
                print(answer, end=':')
                print(score, end=';')
        except:
            # if there are just 2 classes
            pass

        return labels[top_k[0]]



def parse_labels_to_readable_labels(labels, lines, model_to_labels_path):
    for line in lines:
        class_id, class_name = line.split(b'\t')
        labels[class_id.decode()] = class_name.decode().split('\n')[0]
    inv_labels = dict()
    with open(model_to_labels_path, 'rb') as m2l:
        entryset = m2l.read().split(b'entry {')
        for entry in entryset:
            if b'#' not in entry:
                _, class_, class_id = entry.split(b': ')
                class_ = class_[:class_.find(b'\n')].decode()
                class_id = class_id[:class_id.find(b'\n')].split(b'"')[1].decode()
                inv_labels[class_id] = class_
    for class_id in labels.copy():
        if class_id in inv_labels:
            labels[int(inv_labels[class_id])] = labels[class_id]


if __name__ == '__main__':
    if len(sys.argv) == 2:
        run_inference_on_image(image_path=sys.argv[1])
    elif len(sys.argv) == 4:
        run_inference_on_image(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5:
        run_inference_on_image(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 6:
        run_inference_on_image(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

