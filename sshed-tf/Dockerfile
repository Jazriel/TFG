FROM gcr.io/tensorflow/tensorflow:latest-devel-py3

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd

COPY ./id_rsa.pub .
RUN mkdir -p ~/.ssh && cat id_rsa.pub >> ~/.ssh/authorized_keys

RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Directory for images

RUN mkdir -p /images
RUN mkdir -p /datasets
RUN mkdir -p /scripts
RUN mkdir -p /graphs
RUN mkdir -p /bottlenecks

COPY ./run_inference.py /scripts/
COPY ./fit_dataset.py /scripts/
COPY ./maybe_start_retrain.py /scripts/

COPY ./inceptionv3_model.pb /graphs/
COPY ./inceptionv3_labels.txt /graphs/
COPY ./inceptionv3_label_map_proto.pbtxt /graphs/


# SSH login fix. Otherwise user is kicked off after login (Docker doc)
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

WORKDIR /


EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]