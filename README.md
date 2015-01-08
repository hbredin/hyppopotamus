# Hyper-parameter tuning

## On a SGE cluster

#### Step 1 - Run MongoDB

```
$ mongod -v -f ~/mongo/mongodb.conf
```

See file `mongodb.conf` for an example.

Below, we assume that Mongo is ran on host `$MONGO_HOST`.


#### Step 2 - Write experiment 

See file `test/experiment.py` for an example on optimizing `sin(x)`.

Below, we assume that the file is `$EXPERIMENT_DIRECTORY/experiment.py`.  
Note that it is mandatory that the file is called `experiment.py`.

#### Step 3 - Run master job

This job is in charge of choosing the best set of hyper-parameters to evaluate.

```
qsub -V -b y \
     `which python` /absolute/path/to/tune.py \
     --mongo=$MONGO_HOST:27017 $EXPERIMENT_DIRECTORY
```

#### Step 4 - Run worker jobs

These jobs are in charge of evaluating hyper-parameters selected by master job.

```
$ export N_WORKERS=10  # number of workers
$ cd $EXPERIMENT_DIRECTORY
$ qsub -V -b y \
       -cwd \
       -t 1-$N_WORKERS \
       `which hyperopt-mongo-worker` --mongo=$MONGO_HOST:27017/sin
```