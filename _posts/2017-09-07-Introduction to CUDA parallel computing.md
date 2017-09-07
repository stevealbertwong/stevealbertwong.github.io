
##

## Parallel Communication Pattern
```

```
* map tasks(threads in cuda) and memory => data elements, entries, matrix, pixel, same function on each piece of data, one to one correspondence
* gather => map multiple data elements into each thread to one memory location e.g. set multiple neighbour pixels to blur 
* scatter: opposite of gather => each thread take single data element to multiple memory location e.g. each pixel write a fraction of its value to neighbouring pixel => several threads attempting to write to the same memory at same time
* stencil
* transpose:

* reduce
* scale/scan

## CUDA hello world
```
```
CUDA guarantee:
1. All threads inside thread block running a kernel finishes before running the next kernl
2. All threads inside thread block run on the same SM at the same time

## Local Memory vs Shared Memory vs Global Memory vs Host Memory
GPU memory:
* Local Memory: private to that thread, even same block has their own copy of local variable in local memory
* Share Memory(per thread block): shared only to all threads inside thread block
* Global Memory: accessible to any threads in entire system or any kernels.
CPU's memory
* Host Memory: CPU's memory

A programmer's job is to divide his program into smaller computation for GPU to allocate to different Streaming Multiprocessor to compute.
Usually CPU thread loads data from host memory to global memory when launches its work on GPU.

## Memory Model

