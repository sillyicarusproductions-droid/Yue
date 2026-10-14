import pytest
import torch
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from yue2.protocol import Sampling, EOD, ABC_END, MUSIC_END, CODEC_OFFSET, CODEC_SIZE, VOCAB_SIZE
from yue2.sampling import generate_tokens


def native_model():
    torch.set_num_threads(1)
    torch.manual_seed(317)
    return YuE2ForCausalLM(
        YuE2Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=16,
            max_latent_frames=16,
        )
    ).eval()


@pytest.mark.parametrize("phase", ["abc", "semantic"])
def test_greedy_nonfinite_model_logits_never_become_tokens(phase):
    model = native_model()
    with torch.no_grad():
        model.lm_head.weight.fill_(float("nan"))
    events = []
    config = Sampling(temperature=0, min_tokens=0, max_tokens=2)
    with pytest.raises(FloatingPointError, match="finite"):
        generate_tokens(
            model,
            [1, 2],
            config,
            41,
            phase,
            use_cuda_graph=False,
            on_token=lambda stage, token: events.append(token),
        )
    assert events == []


@pytest.mark.parametrize("phase", ["abc", "semantic"])
def test_valid_greedy_model_preserves_phase_vocabulary_and_repeatability(phase):
    model = native_model()
    config = Sampling(temperature=0, min_tokens=0, max_tokens=2)
    first, _, truncated = generate_tokens(model, [1, 2], config, 41, phase, use_cuda_graph=False)
    second, _, again = generate_tokens(model, [1, 2], config, 41, phase, use_cuda_graph=False)
    assert first == second and truncated == again
    assert first
    if phase == "abc":
        assert all(0 <= token < EOD for token in first)
    else:
        assert all(CODEC_OFFSET <= token < CODEC_OFFSET + CODEC_SIZE for token in first)


def test_compiled_decode_step_is_owned_by_the_model(monkeypatch):
    """Dropping a model must release its compiled decode callable: the cache
    lives on the model, not in a global that strongly references every model."""
    import gc
    import weakref
    from yue2 import sampling

    calls = []
    monkeypatch.setattr(sampling.torch, "compile",
                        lambda fn, dynamic=None: (calls.append(1), fn)[1])

    models = [torch.nn.Linear(2, 2) for _ in range(2)]
    steps = [sampling._ensure_decode_step(m) for m in models]
    assert len(calls) == 2
    # Each model owns exactly one step; a second lookup reuses it.
    assert sampling._ensure_decode_step(models[0]) is steps[0]
    refs = [weakref.ref(m) for m in models]
    # The step is a plain callable: attaching it must not register a child
    # module of the model it wraps (that cycle breaks model.to() traversal).
    for m in models:
        assert "yue2_decode_step" not in dict(m.named_modules())
        m.to(torch.float64)  # must not recurse
    del models, steps, m
    gc.collect()
    assert all(ref() is None for ref in refs), "model retained by compiled decode cache"
    assert not hasattr(sampling, "_decode_steps")
