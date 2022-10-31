#!/usr/bin/env python3

import threading

_cache = {}                   #: Actual cache stored as global variable
_writeLock = threading.Lock() #: Write lock for access cache

def cacheMakeKey(purpose, path):
    '''Helper function for create a key from the "path" to source and "purpose" of using resource
    Parameters:
        purpose(str): arbitarily string
        path(str): arbitarily string
    Returns:
        A tuple with strings purpose and path 
    '''
    return  (purpose, path)

def cacheItemThreadSafe(key, value):
    '''Method to cache item in a thread safe manner
    Parameters:
        key: key to use store object in the thread safe cache
        value: value to use store object in the thread safe cache assosiated with the key
    '''
    _writeLock.acquire()
    _cache[key] = value
    _writeLock.release()

def cacheItemThreadUnsafe(key, value):
    '''Cache item in a not thread safe way. Method should be called carefully with a-priorti knowledge that in that moment there is only a single thread that access it.
    Parameters:
        key: key to use store object in the thread safe cache
        value: value to use store object in the thread safe cache assosiated with the key
    '''
    _cache[key] = value

def cacheGetItem(key):
    '''Obtained cached item
    Parameters:
        key: key to use store object in the thread safe cache
    Returns:
        value to use store object in the thread safe cache assosiated with the key
    '''
    return _cache.get(key)

def cacheHasItem(key):
    '''Check that caches has key
    Parameters:
        key: key to use store object in the thread safe cache
    Returns:
        Status that item with a specific key is presetned in the cache
    '''
    return key in _cache

def test_cachestorage():
    cacheMakeKey("save", "path")
    assert cacheMakeKey("save", "path") == cacheMakeKey("save", "path")
    assert cacheMakeKey("load", "path") != cacheMakeKey("load", "path2")
    assert cacheHasItem(cacheMakeKey("s", "path")) == False

    cacheItemThreadUnsafe(cacheMakeKey("s", "path"), 123)
    assert cacheHasItem(cacheMakeKey("s", "path")) == True
    assert cacheGetItem(cacheMakeKey("s", "path")) == 123

    cacheItemThreadSafe(cacheMakeKey("l", "path"), 1234)
    assert cacheHasItem(cacheMakeKey("l", "path")) == True
    assert cacheGetItem(cacheMakeKey("l", "path")) == 1234
