# Python multi-threading and multi-processing

## Multi-Threading

``` python
import threading
import time

# check current thread, usually there is a main thread and multiple child threads
threading.active_count()
threading.enumerate()
threading.current_thread()

def T1_job():
    print('\nChild thread T1 starts: ')
    print(threading.current_thread())
    print('\nActive threads count = %s' % threading.active_count())
    for i in range(10):
        time.sleep(0.1)
    print('\nT1 finished.')

def T2_job():
    print('\nChild thread T2 starts: ')
    print(threading.current_thread())
    print('\nActive threads count = %s' % threading.active_count())
    print('\nT2 finished.')

def main():
    # create a new thread
    thread_1 = threading.Thread(target=T1_job)
    thread_2 = threading.Thread(target=T2_job)
    thread_1.start()
    thread_2.start()
    thread_2.join() # blocks the main thead until thread_2 terminates
    thread_1.join()
    print('all done')

if __name__ == '__main__':
    main()
```

### Use Queue to share data between threads

``` python
import threading
import time

from queue import Queue

# calculate sqaure of each element in list l and save the result in queue q
def job(l, q):
    for i in range(len(l)):
        l[i] = l[i]**2
    q.put(l)

def multithreading():
    q = Queue()
    threads = []
    data = [[1,2,3],[3,4,5],[4,4,4],[5,5,5]]

    for i in range(4):
        t = threading.Thread(target=job, args=(data[i],q))
        t.start()
        threads.append(t) # append each thread to threads list

    for thread in threads:
        thread.join() # block main thread for all children threads done

    results = []
    for _ in range(4):
        results.append(q.get())
    print(results)

if __name___=='__main__':
    multithreading()
```

> Notice: Python GIL is not very CPU effective, but if your threads are doing lots of I/O rather than computing, it is safe and efficient to use thousands of threads.

### Lock shared memory

If you're sharing a global variable between threads and to keep them safe, you need to use `lock.acquire()` before accessing the global memory and release it using `lock.release()` after update the global memory.

## Multi-Processing

``` python
import multiprocessing as mp
import threading as td

def job(a,d):
    print('aaaaa')

if __name__=='__main__':
    t1 = td.Thread(target=job,args=(1,2))
    p1 = mp.Process(target=job,args=(1,2))

    t1.start()
    p1.start()

    t1.join()
    p1.join()
```

### Save multiple process return result in Queue

``` python
def job(q):
    res=0
    for i in range(1000):
        res+=i+i**2+i**3
    q.put(res)    #queue

if __name__=='__main__':
    q = mp.Queue()
    p1 = mp.Process(target=job,args=(q,))
    p2 = mp.Process(target=job,args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()

    print(res1+res2)
```

### Process Pool

``` python
import multiprocessing as mp

def job(x):
    return x*x

def multicore():
    pool = mp.Pool()
    res = pool.map(job, range(10)) # use pool to allocate job and get result
    print(res)
    res = pool.apply_async(job, (2,))
    print(res.get())

if __name__ == '__main__':
    multicore()
```

``` python
import types

# print all imported modules
def imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            print(val.__name__)

imports()

# another way to show module name and from
import sys
modulenames = set(sys.modules) & set(globals())
allmodules = [sys.modules[name] for name in modulenames]
```
