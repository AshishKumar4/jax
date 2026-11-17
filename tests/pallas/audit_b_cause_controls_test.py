# Temporary causal instrumentation for the fork audit; no product edits.
import importlib.util
from pathlib import Path
from unittest import mock

from absl.testing import absltest
from jax._src import core
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import lowering
from jax._src.pallas.mosaic_gpu import primitives
from jax.experimental import mosaic


def load(name):
  spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class CauseControlsTest(absltest.TestCase):

  def test_waiting_foreign_execution_restores_progress(self):
    module = load('gpu_pallas_interpret_test')
    run = module.subprocess.run

    def session_gated(args, **kwargs):
      args = list(args)
      before = 'def initialize(threads):\n'
      after = (
          'def initialize(threads):\n'
          '  if threads == 2:\n'
          '    a_done.wait()\n'
      )
      self.assertEqual(args[2].count(before), 1)
      args[2] = args[2].replace(before, after)
      return run(args, **kwargs)

    with mock.patch.object(module.subprocess, 'run', session_gated):
      method = 'test_two_device_execution_can_finish_overlapping_executions'
      case = module.InterpretExecutionGateTest(method)
      getattr(case, method)()

  def test_cta_warp_context_guard_restores_refusal(self):
    module = load('mosaic_gpu_test')
    rule = primitives._barrier_arrive_and_wait_lowering

    def guarded(ctx, barrier, *args, **kwargs):
      if (isinstance(barrier, mosaic.gpu.CtaBarrierRef)
          and ctx.module_ctx.primitive_semantics == gpu_core.PrimitiveSemantics.Warp):
        raise NotImplementedError(
            'Arriving on a CTA barrier is not supported in a warp context')
      return rule(ctx, barrier, *args, **kwargs)

    for semantics, suffix in (
        (gpu_core.LANExWARP_SEMANTICS, 'lane'),
        (gpu_core.WGxWARP_SEMANTICS, 'warpgroup'),
    ):
      with mock.patch.dict(
          lowering.mosaic_lowering_rules[semantics],
          {primitives.barrier_arrive_and_wait_p: guarded},
      ):
        method = (
            'test_conditional_warp_cannot_arrive_on_cta_barrier_arrive_and_wait_'
            + suffix)
        case = module.CtaBarrierLoweringTest(method)
        getattr(case, method)()

  def test_declaring_meeting_barriers_restores_resource_equality(self):
    module = load('mosaic_gpu_test')
    original = module.plgpu.emit_pipeline

    def emit(*args, **kwargs):
      pipeline = original(*args, **kwargs)
      get_allocations = pipeline.get_allocations

      def declared(*refs):
        result = get_allocations(*refs)
        axis = kwargs.get('thread_axis')
        threads = 1 if axis is None else core.get_axis_env().axis_size(axis)
        result['meet_ref'] = (
            None if threads == 1
            else gpu_core.Barrier(num_arrivals=threads, num_barriers=2))
        return result

      pipeline.get_allocations = declared
      return pipeline

    with mock.patch.object(module.plgpu, 'emit_pipeline', emit):
      for suffix in ('0', '1'):
        method = 'test_collective_pipeline_declares_its_meeting_barriers' + suffix
        case = module.PipelineAllocationsTest(method)
        getattr(case, method)()
