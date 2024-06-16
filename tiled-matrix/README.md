# Tiled matrix multiplication

## Description

This directory contains a reference implementation of tiled matrix multiplication and three AI created implementations. Input matrices are read from files. Output is put into two separate files - one for GPU and one for CPU for checking if the results are correct.

## How to run

Change to an implementations direcotory, compile with nvcc and run the executable:

```
$ cd <implementation-directory>
$ nvcc <kernel-name>.cu
```