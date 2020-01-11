# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 20:31:33 2019

@author: Santi
"""

import queuelib

q = FifoDiskQueue("queuefile")

for i in range(100):
    q.put(i)
    print(q)
