# PySpark ML algorithms on Hadoop file system and yarn resource manager in docker container in Multi-Node Cluster mode

## How to Run
- Go to your terminal.
- Run the following script
	```
	# Here, N = number of slave nodes to create (default value is 3).
	. ./restart-all.sh   N

	```
- Install pip3 inside node master and on all slaves 

	```
	apt update
	apt install python3-pip
	
	```
- Install numpy in all nodes

	```
	pip3 install numpy

	```
- Install scikit learn on all nodes

	```
	pip3 install scikit-learn

	```

- Start HDFS and yarn resource manager in the master

	```
	start-all.sh

	```

- Upload files from local to docker master

	```
	#On your terminal
	docker cp file_name testbed-master:/directory

	```

- Upload files from docker master fils system on hdfs

	```
	hdfs dfs -put log2.csv /

	```

- Spark submit in cluster mode

	```
	spark-submit --master yarn --deploy-mode cluster --num-executors 1 --driver-memory 2g --executor-memory 512m --executor-cores 1 script.py 100

	```









