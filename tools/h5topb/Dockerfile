FROM oraclelinux:7-slim
RUN yum install -y yum-utils
RUN yum-config-manager --enable ol7_developer_EPEL
RUN yum install -y python36
RUN python3.6 -m venv py36env
RUN source py36env/bin/activate && pip install tensorflow==1.9.0
RUN source py36env/bin/activate && pip install h5py
COPY convert.py /convert.py
COPY h5topb.sh /h5topb.sh
ENTRYPOINT ["/h5topb.sh"]
