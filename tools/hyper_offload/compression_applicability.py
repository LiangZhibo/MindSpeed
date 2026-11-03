"""Diagnostic-only calibration on owned copies of real warmup activations.

This never selects a training policy. All instrumented runs are ineligible for
performance scoring. First-warmup copies are explicitly synchronized before
cloning; no alias to recycled Host-pool memory survives the capture. Bounded
size strata favor large observed traffic without any model or tensor-size rule.
"""
from dataclasses import asdict
import hashlib
import time

import torch
from torch.utils._python_dispatch import _disable_current_modes


class SampleBudget:
    def __init__(self, max_bytes=512*1024**2, max_groups=3, per_group=2):
        if any(isinstance(x,bool) or not isinstance(x,int) or x < 1
               for x in (max_bytes,max_groups,per_group)):
            raise ValueError('positive sample limits required')
        self.max_bytes,self.max_groups,self.per_group=max_bytes,max_groups,per_group
        self.items=[]
        self.bytes=0
        self.peak_bytes=0

    def admit(self, size):
        if isinstance(size,bool) or not isinstance(size,int) or size < 2 or size%2 or size>self.max_bytes:
            return None
        group=(size-1).bit_length()
        if sum(x['group']==group for x in self.items)>=self.per_group:
            return None
        keep=list(self.items)
        while (sum(x['bytes'] for x in keep)+size>self.max_bytes
               or len({x['group'] for x in keep}|{group})>self.max_groups):
            smaller=[x for x in keep if x['group']<group]
            if not smaller:return None
            victim=min(smaller,key=lambda x:(x['group'],x['bytes']))
            keep=[x for x in keep if x is not victim]
        # Release evicted owned snapshots before allocating the new one.
        self.items=keep
        self.bytes=sum(x['bytes'] for x in keep)
        return group

    def retain(self, group, data, metadata):
        size=data.numel()*data.element_size()
        if (size-1).bit_length()!=group or self.bytes+size>self.max_bytes:
            raise ValueError('sample reservation does not match owned data')
        self.items.append(dict(group=group,bytes=size,data=data,metadata=metadata))
        self.bytes+=size
        self.peak_bytes=max(self.peak_bytes,self.bytes)


class WarmupSamples:
    def __init__(self, **limits):
        self.budget=SampleBudget(**limits)
        self.closed=False
        self.capture_seconds=0.0
        self.copy_observations=0
        self._original=None

    def install(self):
        from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager
        if self._original is not None:raise RuntimeError('sampler already installed')
        original=ResidencyManager.copy_d2h
        self._original=original
        def copy(manager,sid):
            buffer=manager._residency.get(sid)
            fresh=buffer is not None and buffer.host_buffer is None
            result=original(manager,sid)
            if fresh and not self.closed:
                self.copy_observations+=1
                buffer=manager._residency[sid]
                if buffer.storage_dtype==torch.bfloat16 and buffer.host_encoding is None:
                    group=(self.budget.admit(buffer.logical_nbytes)
                           if manager._host_pool.total_allocated+self.budget.bytes+buffer.logical_nbytes
                           <=manager._host_pool.max_host_bytes else None)
                    if group is not None:
                        started=time.perf_counter()
                        with _disable_current_modes():
                            if buffer.host_event is not None:buffer.host_event.synchronize()
                            owned=buffer.host_buffer.clone()
                        node=manager._transfer_node
                        self.budget.retain(group,owned,dict(storage_id=sid,node=asdict(node) if node else None))
                        self.capture_seconds+=time.perf_counter()-started
            return result
        ResidencyManager.copy_d2h=copy

    def close(self):
        if self.closed:return
        self.closed=True
        if self._original is not None:
            from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager
            ResidencyManager.copy_d2h=self._original
            self._original=None


def calibrate(samples, *, library, device, device_limit, host_limit, training_host_bytes,
              repetitions=3):
    """Isolated complete raw/codec roundtrips; encode/DMA/decode are inclusive.

    CPU sample cloning and this calibration perturb execution, so timings only
    diagnose applicability. They are not steady training or live tuning costs.
    Full-step candidate tests remain necessary for overlap/queue interference.
    """
    from mindspeed.core.memory.compression import SplitZipCodec
    from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager
    from mindspeed.core.memory.hyper_offload.runtime.pinned_memory import _align_to_bucket
    if not samples.closed:raise ValueError('close warmup sampler before calibration')
    if repetitions<1:raise ValueError('positive measured repetitions required')
    started=time.perf_counter();rows=[]
    # A bounded codebook sample covers beginning/middle/end of every owned
    # snapshot. No synthetic distribution or assumed codebook is substituted.
    portions=[]
    for item in samples.budget.items:
        data=item['data'];count=min(data.numel(),64*1024)//2*2
        for offset in (0,((data.numel()-count)//4)*2,data.numel()-count):
            portions.append(data[offset:offset+count].view(torch.bfloat16))
    if not portions:
        return dict(status='no_supported_samples',rows=[],capture_seconds=samples.capture_seconds)
    estimated_book_bytes=sum(x.numel()*x.element_size() for x in portions)
    if training_host_bytes+samples.budget.bytes+2*estimated_book_bytes+2*1024**2>host_limit:
        return dict(status='skipped_host_capacity',rows=[],capture_seconds=samples.capture_seconds)
    if torch.npu.memory_allocated(device)+4*1024**2>device_limit:
        return dict(status='skipped_device_capacity',rows=[],capture_seconds=samples.capture_seconds)
    codebook_sample=torch.cat(portions)
    codec=SplitZipCodec(library,sample=codebook_sample)
    book_bytes=codebook_sample.numel()*codebook_sample.element_size()
    torch.npu.synchronize(device)
    initial_allocated=torch.npu.memory_allocated(device)
    torch.npu.reset_peak_memory_stats(device)
    try:
        prepare=time.perf_counter();codec.prepare(device);torch.npu.synchronize(device)
        prepare_seconds=time.perf_counter()-prepare
        for item in sorted(samples.budget.items,key=lambda x:x['bytes']):
            size=item['bytes'];padded=((size+16383)//16384)*16384
            # Source, restored output, packet, staging/metadata and allocator
            # margin are reserved conservatively, without ratio discounts.
            device_bound=6*padded+4*1024**2
            pool_limit=2*_align_to_bucket(padded+2048)
            host_bound=samples.budget.bytes+2*book_bytes+2*pool_limit+size+2*1024**2
            row=dict(bytes=size,group=item['group'],metadata=item['metadata'],
                     source_sha256=hashlib.sha256(memoryview(item['data'].numpy()).cast('B')).hexdigest(),
                     required_extra_device_bytes=device_bound,required_extra_host_bytes=host_bound)
            rows.append(row)
            if (torch.npu.memory_allocated(device)+device_bound>device_limit
                    or training_host_bytes+host_bound>host_limit):
                row['status']='skipped_capacity';continue
            source=item['data'].view(torch.bfloat16).to(device)
            torch.npu.synchronize(device)
            managers={name:ResidencyManager(pool_limit,codec=value)
                      for name,value in [('raw',None),('codec',codec)]}
            def roundtrip(name):
                manager=managers[name];buffer=manager.bind(1,source)
                torch.npu.synchronize(device)
                before=time.perf_counter();manager.copy_d2h(1)
                if buffer.host_event is not None:buffer.host_event.synchronize()
                d2h=time.perf_counter()-before
                encoded=buffer.host_encoding
                packet=(encoded.metadata.native_packet.stored_bytes() if encoded is not None
                        else buffer.host_buffer.numel())
                manager.release_device(1)
                before=time.perf_counter();manager.copy_h2d(1)
                if buffer.device_event is not None:buffer.device_event.synchronize()
                h2d=time.perf_counter()-before
                restored=buffer.device_buffer
                exact=torch.equal(restored.cpu(),item['data'])
                if not exact:raise AssertionError('actual activation codec roundtrip mismatch')
                row_peak=torch.npu.max_memory_allocated(device)
                if row_peak>device_limit:raise RuntimeError('calibration device capacity exceeded')
                manager.clear_runtime(set())
                del restored,encoded,buffer
                return dict(d2h_inclusive_seconds=d2h,h2d_inclusive_seconds=h2d,
                            roundtrip_seconds=d2h+h2d,packet_bytes=packet,lossless_exact=exact,
                            peak_allocated_bytes=row_peak)
            row['warmup']={name:roundtrip(name) for name in ['raw','codec']}
            row['measurements']=[]
            for index in range(repetitions):
                order=['raw','codec'] if index%2==0 else ['codec','raw']
                row['measurements'].append(dict(index=index,order=order,
                                               results={name:roundtrip(name) for name in order}))
            row['status']='complete'
            for manager in managers.values():manager.sync_all_transfers();manager.clear_runtime(set())
            del source,managers,manager
        return dict(status='complete',rows=rows,capture_seconds=samples.capture_seconds,
            copy_observations=samples.copy_observations,owned_sample_bytes=samples.budget.bytes,
            sample_peak_bytes=samples.budget.peak_bytes,codebook_sample_bytes=book_bytes,
            codebook_sha256=hashlib.sha256(memoryview(codebook_sample.view(torch.uint8).numpy()).cast('B')).hexdigest(),
            codec_base=codec._base,codec_stats=dict(codec.stats),prepare_seconds=prepare_seconds,
            calibration_seconds=time.perf_counter()-started,initial_allocated_bytes=initial_allocated,
            peak_allocated_bytes=torch.npu.max_memory_allocated(device),
            performance_usable=False,training_policy_changed=False,
            scope='Owned real first-warmup activations; bounded size strata and first two examples per group are not distribution coverage; isolated serialized inclusive raw/codec costs cannot establish training overlap or policy speedup')
    finally:
        codec.close()
