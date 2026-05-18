import pytest
import torch
import triton
import triton.language as tl


@pytest.fixture
def fresh_knobs_local():
    from triton._internal_testing import _fresh_knobs_impl

    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()


def _skip_unless_npu():
    try:
        import torch_npu  # noqa: F401
    except Exception as e:
        pytest.skip(f'torch_npu unavailable: {e}')


def test_costmodel_mode_routes_to_costmodel_backend(monkeypatch, fresh_knobs_local):
    _skip_unless_npu()

    triton.knobs.autotuning.cache = False
    calls = {'costmodel': 0}

    from triton.runtime.autotuner import Autotuner

    def _stub_costmodel(self, *args, pruned_configs, key, **kwargs):
        calls['costmodel'] += 1
        self.cache[key] = pruned_configs[0]
        self.configs_timings = {cfg: float(i + 1) for i, cfg in enumerate(pruned_configs)}
        self.bench_time = 0.0

    monkeypatch.setattr(Autotuner, '_costmodel_bench', _stub_costmodel, raising=True)

    @triton.autotune(
        configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 256}),
            triton.Config(kwargs={'BLOCK_SIZE': 512}),
        ],
        key=['n_elements'],
    )
    @triton.jit
    def _kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    device = 'npu'
    x = torch.rand(1024, device=device)
    y = torch.rand(1024, device=device)
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(out.numel(), meta['BLOCK_SIZE']),)
    _kernel[grid](x, y, out, out.numel(), enable_costmodel_backend=True)

    assert calls['costmodel'] >= 1, 'enable_costmodel_backend=True should route to _costmodel_bench'


def test_without_costmodel_mode_does_not_route_to_costmodel_backend(monkeypatch, fresh_knobs_local):
    _skip_unless_npu()

    triton.knobs.autotuning.cache = False
    calls = {'costmodel': 0}

    from triton.runtime.autotuner import Autotuner

    def _stub_costmodel(self, *args, pruned_configs, key, **kwargs):
        calls['costmodel'] += 1
        self.cache[key] = pruned_configs[0]
        self.configs_timings = {cfg: float(i + 1) for i, cfg in enumerate(pruned_configs)}
        self.bench_time = 0.0

    monkeypatch.setattr(Autotuner, '_costmodel_bench', _stub_costmodel, raising=True)

    @triton.autotune(
        configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 256}),
            triton.Config(kwargs={'BLOCK_SIZE': 512}),
        ],
        key=['n_elements'],
    )
    @triton.jit
    def _kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    device = 'npu'
    x = torch.rand(1024, device=device)
    y = torch.rand(1024, device=device)
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(out.numel(), meta['BLOCK_SIZE']),)
    _kernel[grid](x, y, out, out.numel(), enable_costmodel_backend=False)

    assert calls['costmodel'] == 0, 'enable_costmodel_backend=False should not route to _costmodel_bench'
