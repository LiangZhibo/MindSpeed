from dataclasses import replace

import pytest

from mindspeed.core.memory.hyper_offload.ir.trace import AccessKind as K, ActivationTrace, StorageAccess, TraceOp
from mindspeed.core.memory.hyper_offload.tuning.trace_regions import region_costs


def fixture():
    def op(i, name, accesses):
        return TraceOp(name, 1., [StorageAccess(i, sid, kind) for sid, kind in accesses])
    trace = ActivationTrace([
        op(0,'input',[(1,K.WRITE)]),
        op(1,'expand',[(1,K.READ),(2,K.WRITE)]),
        op(2,'nonlinearity',[(2,K.READ),(3,K.WRITE)]),
        op(3,'view',[(3,K.READ),(3,K.WRITE)]),
        op(4,'loss',[(3,K.READ)]),
        op(5,'backward',[(1,K.READ),(2,K.READ),(3,K.READ)]),
    ], {1:10,2:200,3:20})
    report=dict(forward_end_op=4,truncated=False,regions=[dict(name='anything.region',structure=('custom',),
        input_shapes=(),input_boundary_bytes=10,output_boundary_bytes=20,input_storage_ids=(1,),
        output_storage_ids=(3,),first_op=1,last_op=3,complete=True,opaque=False,inclusive_interval_ms=9000.)])
    return trace, report


def test_saved_internal_versions_not_boundaries_or_inclusive_interval():
    trace, report=fixture()
    cost,=region_costs(trace,report)
    assert cost.saved_bytes==200 and cost.compute_ms==3.
    assert cost.saved_versions[0].first_backward_read==5
    # Renaming the adapter changes identity only, never the measurements.
    report['regions'][0]['name']='different.network'
    other,=region_costs(trace,report)
    assert replace(other,name=cost.name)==cost


def test_overwritten_version_not_mistaken_for_saved_activation():
    trace, report=fixture()
    trace.ops[4].accesses.append(StorageAccess(4,2,K.WRITE))
    cost,=region_costs(trace,report)
    assert cost.saved_bytes==0


def test_mutating_existing_storage_is_not_assumed_checkpoint_safe():
    trace, report=fixture()
    trace.ops[2].accesses.append(StorageAccess(2,2,K.WRITE))
    cost,=region_costs(trace,report)
    assert cost.saved_bytes==0


def test_missing_opaque_and_collective_evidence_never_becomes_zero_cost():
    trace, report=fixture()
    for field in ('opaque','truncated'):
        target=report if field=='truncated' else report['regions'][0]
        target[field]=True
        assert region_costs(trace,report)==()
        target[field]=False
    trace.ops[2].name='all_gather_into_tensor'
    assert region_costs(trace,report)==()


def test_retained_storage_and_sparse_positions_are_not_attributed():
    trace, report=fixture()
    trace.retained_sids.add(2)
    assert region_costs(trace,report)[0].saved_bytes==0
    trace.ops[2].accesses[0].op_id=123
    with pytest.raises(ValueError,match='dense'):
        region_costs(trace,report)


def test_single_storage_inplace_op_is_not_assumed_to_be_a_view():
    trace, report=fixture()
    trace.ops[3]=TraceOp('add_.Tensor',1.,[StorageAccess(3,2,K.READ),StorageAccess(3,2,K.WRITE)])
    assert region_costs(trace,report)[0].saved_bytes==0


def test_last_backward_use_and_compute_windows_exclude_collective_time():
    trace, report=fixture()
    trace.ops.extend([
        TraceOp('all_reduce',100.,[]),
        TraceOp('backward_late',2.,[StorageAccess(7,2,K.READ)]),
    ])
    cost,=region_costs(trace,report)
    assert cost.saved_versions[0].first_backward_read==5
    assert cost.saved_versions[0].last_backward_read==7
    assert cost.forward_save_window_ms==2.
    assert cost.backward_save_window_ms==1.  # No collective or late consumer time.
    assert cost.working_set_bytes==230  # All distinct inputs/outputs, not cumulative writes.
