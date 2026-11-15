"""Bounded, non-persistent caches for deterministic, parameter-free constants."""
from collections import OrderedDict
from threading import RLock
import os
import torch

_items=OrderedDict()
_bytes=0
_lock=RLock()


def clear():
    global _bytes
    with _lock:
        _items.clear();_bytes=0


def get(key,builder,device):
    global _bytes
    # Keep the compiled path free of Python cache mutation and locks.
    if torch.compiler.is_compiling():return builder()
    budget=int(os.environ.get('MIRAGE_CONSTANT_CACHE_MB','32'))*1024**2
    if budget<0:raise ValueError('MIRAGE_CONSTANT_CACHE_MB must be nonnegative')
    device=torch.device(device)
    if device.type=='cuda' and device.index is None:
        device=torch.device('cuda',torch.cuda.current_device())
    key=(device,key)
    with _lock:
        while _items and _bytes>budget:
            _,(_,_,_,size)=_items.popitem(last=False);_bytes-=size
        entry=_items.get(key) if budget else None
        stream=torch.cuda.current_stream(device) if device.type=='cuda' else None
        if entry is None:
            # An inference preview may populate this cache before training.
            # Normal tensors can safely be saved for a later backward pass.
            with torch.inference_mode(False),torch.no_grad():value=builder()
            tensors=value if isinstance(value,tuple) else (value,)
            size=sum(t.numel()*t.element_size() for t in tensors)
            event=None;owner=None
            if stream is not None:
                event=torch.cuda.Event();event.record(stream);owner=stream.cuda_stream
            if budget and size<=budget:
                while _items and _bytes+size>budget:
                    _,(_,_,_,old_size)=_items.popitem(last=False);_bytes-=old_size
                _items[key]=(value,event,owner,size);_bytes+=size
            return value
        _items.move_to_end(key)
        value,event,owner,_=entry
        if stream is not None and stream.cuda_stream!=owner:
            stream.wait_event(event)
            for tensor in (value if isinstance(value,tuple) else (value,)):tensor.record_stream(stream)
        return value
