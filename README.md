# tfpd

Simple tf application to see how to use the `tf.profiler` and `tfdbg` with the high level APIs.

## Requirements

To visualize the tracing, you need to use the [profiler-ui](https://github.com/tensorflow/profiler-ui) tool.
Alternatively, you can convert it into a chrome tracing compatible format, but I prefer to use `profiler-ui`.
There are zero extra steps.

## Generating data

To test higher workloads, you can generate as many examples as you would like using the `tfpd.data` module.

```bash
$ python -m tfpd.data
``` 

For example, you can generate 10M records in 500 partitions. Then profile using one reading thread and multiple parsing
threads; shuffling vs no shuffling.

With that, you can use define a deeper network, e.g. `--num-layers=100`, and study the profile tracing.


## Training/Profiling/Debugging

To see what options are available, run:

```
$ python -m tfpd.task --help
```

If you wish to use the [sbin/run.sh](sbin/run.sh) script, you may.
It will run training and testing, then save the output to the path `$HOME/fs/models/tfpd/basic/{current-timestamp}`.
So each execution should give you a new `model-dir`.
You can pass the script any parameter you would to the python module.
The following also works:

```bash
./sbin/run.sh --help
```


### Profiling

With the high level APIs, you don't usually have access to the `tf.Session`.
However, it's fairly easy to set it up. You just wrap your code inside a profiler
context. See the main function in [tfpd/task.py](tfpd/task.py)

Note:

  - Profiling slows down performance in general.
  - The profiler here uses sampling to trace performance, so the results should be read with a grain of salt.
  E.g. very fast operations that occur at a high frequency may not be attributed their proportionate runtime.
  
Each execution places an output file called `profile.pb` on the `model-dir` path.
To open it, from the path of `profiler-ui`, type:

```
python -m ui --profile_context_path=PATH_TO_MODEL_DIR/profile.pb
```


### Debugging


Start by reading the [debugger guide](https://www.tensorflow.org/guide/debugger)

It explains how to set it up, and what you can do with it.
Once on `tfdbg`, you can filter tensors based on pre-defined filters, e.g. `$lt -f has_inf_or_nan`.
It's also possible to setup [custom filters](https://www.tensorflow.org/api_docs/python/tfdbg/DebugDumpDir#find).

We use the `--debug` flag to turn on the debugger.
