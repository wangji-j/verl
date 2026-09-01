import torch
from tensordict import TensorDict

from verl.utils.fsdp_router_trace import FSDPRouterTrace


class DummyRouter(torch.nn.Module):
    def forward(self, hidden_states):
        selected = torch.topk(hidden_states, k=2, dim=-1).indices
        return hidden_states, hidden_states, selected


class DummyMoe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = DummyRouter()

    def forward(self, hidden_states):
        return self.router(hidden_states)


def test_fsdp_router_trace_captures_qwen_style_router_output_on_cpu():
    trace = FSDPRouterTrace()
    model = DummyMoe()

    assert trace.install(model) == 1
    trace.enabled = True
    with trace.capture(True):
        model(torch.tensor([[[0.1, 0.4, 0.3], [0.9, 0.2, 0.1]]]))

    input_ids = torch.nested.nested_tensor([torch.tensor([1, 2])], layout=torch.jagged)
    batch = TensorDict({"input_ids": input_ids}, batch_size=[1])
    routed = trace.consume(batch)

    assert routed is not None
    padded = routed.to_padded_tensor(0)
    assert padded.shape == (1, 2, 1, 2)
    torch.testing.assert_close(padded[0, :, 0], torch.tensor([[1, 2], [0, 1]]))
