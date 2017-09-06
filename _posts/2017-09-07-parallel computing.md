
## Parallel Communication Pattern
* map tasks(threads in cuda) and memory => data elements, entries, matrix, pixel, same function on each piece of data, one to one correspondence
* gather => map multiple data elements into each thread to one memory location e.g. set multiple neighbour pixels to blur 
* scatter: opposite of gather => each thread take single data element to multiple memory location e.g. each pixel write a fraction of its value to neighbouring pixel => several threads attempting to write to the same memory at same time


##

