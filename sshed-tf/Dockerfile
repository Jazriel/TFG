FROM gcr.io/tensorflow/tensorflow:latest-devel-py3

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir -p /var/run/sshd

RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

COPY ./resources/id_rsa.pub .
RUN mkdir -p ~/.ssh && cat id_rsa.pub >> ~/.ssh/authorized_keys

# Directory for images

RUN mkdir -p /images && mkdir -p /datasets && mkdir -p /scripts && mkdir -p /graphs

COPY ./scripts /scripts/
COPY ./graphs /graphs/

# SSH login fix. Otherwise user is kicked off after login (Docker doc)
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

WORKDIR /

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]