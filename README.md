# neuroseg


### resolving TF 2.7.0 issues: wrong CUDA libs

If TF 2.7 doesn't find the correct CUDA libs in an anaconda env it might depend on how libraries are loaded by TF after 2.7

just set ```LD_LIBRARY_PATH``` 

```
export LD_LIBRARY_PATH=/home/phil/anaconda3/envs/neuroseg/lib
```
