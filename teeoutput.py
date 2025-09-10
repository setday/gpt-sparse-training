# -*- coding: utf-8 -*-
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

class TeeOutput:
    def __init__(self, original_std, buffer):
        self.original_std = original_std
        self.buffer = buffer
    
    def write(self, text):
        # Записываем в оригинальный поток (консоль)
        self.original_std.write(text)
        self.original_std.flush()
        # Записываем в буфер для перехвата
        self.buffer.write(text)
    
    def flush(self):
        self.original_std.flush()
        self.buffer.flush()
    
    def __getattr__(self, attr):
        # Проксируем остальные методы оригинальному потоку
        return getattr(self.original_std, attr)