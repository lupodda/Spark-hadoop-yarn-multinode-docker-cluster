# Coding Style: Shell form

# Start from Ubuntu OS image
FROM ubuntu:22.04

# set root user
USER root

# install utilities on up-to-date node
RUN apt-get update 
RUN apt-get -y dist-upgrade
RUN apt-get -y install software-properties-common
RUN add-apt-repository ppa:openjdk-r/ppa
RUN apt-get -y update
RUN apt-get install -y openjdk-11-jdk
RUN apt-get install -y openssh-server wget scala

# set java home
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# setup ssh with no passphrase
RUN ssh-keygen -t rsa -f $HOME/.ssh/id_rsa -P "" \
    && cat $HOME/.ssh/id_rsa.pub >> $HOME/.ssh/authorized_keys

# download & extract & move hadoop & clean up
# TODO: write a way of untarring file to "/usr/local/hadoop" directly
RUN wget https://dlcdn.apache.org/hadoop/common/hadoop-3.3.4/hadoop-3.3.4.tar.gz
RUN tar xfz hadoop-3.3.4.tar.gz
RUN mv hadoop-3.3.4 /usr/local/hadoop 
RUN rm hadoop-3.3.4.tar.gz

# download & extract & move spark & clean up
# TODO: write a way of untarring file to "/usr/local/spark" directly
RUN wget https://dlcdn.apache.org/spark/spark-3.3.1/spark-3.3.1-bin-hadoop3.tgz --no-check-certificate 
RUN mv spark-3.3.1-bin-hadoop3.tgz spark-3.3.1-bin-hadoop3.tar.gz
RUN tar xfz spark-3.3.1-bin-hadoop3.tar.gz 
RUN mv spark-3.3.1-bin-hadoop3 /usr/local/spark
RUN rm spark-3.3.1-bin-hadoop3.tar.gz 

# hadoop environment variables
ENV HADOOP_HOME=/usr/local/hadoop
ENV SPARK_HOME=/usr/local/spark
ENV PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin:$SPARK_HOME:sbin
ENV HDFS_NAMENODE_USER="root"
ENV HDFS_DATANODE_USER="root"
ENV HDFS_SECONDARYNAMENODE_USER="root"
ENV YARN_RESOURCEMANAGER_USER="root"
ENV YARN_NODEMANAGER_USER="root"

# hadoop-store
RUN mkdir -p $HADOOP_HOME/hdfs/namenode 
RUN mkdir -p $HADOOP_HOME/hdfs/datanode

# setup configs - [standalone, pseudo-distributed mode, fully distributed mode]
# NOTE: Directly using COPY/ ADD will NOT work if you are NOT using absolute paths inside the docker image.
# Temporary files: http://refspecs.linuxfoundation.org/FHS_3.0/fhs/ch03s18.html
COPY config/ /tmp/
RUN mv /tmp/ssh_config $HOME/.ssh/config 
RUN mv /tmp/hadoop-env.sh $HADOOP_HOME/etc/hadoop/hadoop-env.sh 
RUN mv /tmp/core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml 
RUN mv /tmp/hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml 
RUN mv /tmp/mapred-site.xml $HADOOP_HOME/etc/hadoop/mapred-site.xml.template 
RUN cp $HADOOP_HOME/etc/hadoop/mapred-site.xml.template $HADOOP_HOME/etc/hadoop/mapred-site.xml 
RUN mv /tmp/yarn-site.xml $HADOOP_HOME/etc/hadoop/yarn-site.xml 
RUN cp /tmp/workers $HADOOP_HOME/etc/hadoop/workers 
RUN mv /tmp/workers $SPARK_HOME/conf/workers 
RUN mv /tmp/spark/spark-env.sh $SPARK_HOME/conf/spark-env.sh 
RUN mv /tmp/spark/log4j.properties $SPARK_HOME/conf/log4j.properties

# Add startup script
ADD scripts/spark-services.sh $HADOOP_HOME/spark-services.sh

# set permissions
RUN chmod 744 -R $HADOOP_HOME

# format namenode
RUN $HADOOP_HOME/bin/hdfs namenode -format

# run hadoop services
ENTRYPOINT service ssh start; cd $SPARK_HOME; bash

