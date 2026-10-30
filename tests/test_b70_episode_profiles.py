"""Episode profile invariants for the Intel upstream release."""
import json
from pathlib import Path
from ai_toolbox_cockpit.catalog import load_model_catalog, load_toolbox_catalog
from ai_toolbox_cockpit.backends.vllm.runner import (
    apply_device_selection, apply_toolbox_policy_overrides, build_server_cmd,
)


def test_b70_episode_int4_uses_graphs_and_embedded_mtp_on_one_or_two_cards():
    catalog = load_toolbox_catalog()
    toolbox = catalog.toolboxes['intel-b70-vllm-upstream']
    base = next(x for x in load_model_catalog().backends['vllm'].entries
                if x['id'] == 'vllm-redhatai-qwen3-8-27b-int4')
    for count, devices in [(1, '1'), (2, '0,1')]:
        policy = apply_device_selection(apply_toolbox_policy_overrides(base, toolbox.backend_config), devices, count)
        command = build_server_cmd(engine='podman', image=toolbox.image,
            engine_args=catalog.runtime_profiles[toolbox.runtime_profile].engine_args,
            model_id=base['repo'], policy=policy, tensor_parallel=count,
            model_directory=Path('/tmp/b70-int4'), speculation='mtp')
        assert '--enforce-eager' not in command
        assert command[command.index('--max-model-len') + 1] == '67840'
        assert command[command.index('--tensor-parallel-size') + 1] == str(count)
        assert f'ZE_AFFINITY_MASK={devices}' in command
        assert '--enable-prefix-caching' in command
        assert json.loads(command[command.index('--speculative-config') + 1]) == {'method': 'mtp', 'num_speculative_tokens': 3}
    scaler = catalog.toolboxes['intel-b70-vllm-scaler']
    scaler_policy = apply_toolbox_policy_overrides(base, scaler.backend_config)
    assert scaler_policy['enforce_eager'] is True
    assert scaler_policy['speculation'] == {}
    assert scaler_policy['valid_tp'] == [1]


def test_flash_atomicchat_identity_and_external_sycl_mtp_do_not_use_generic_flash_defaults():
    import shlex
    from ai_toolbox_cockpit.backends.llama_cpp.config import get_model_config, get_effective_mtp_config, get_mtp_server_args
    model=get_model_config('/models/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf')
    assert model['id']=='llama-atomicchat-qwen3-8-flash-next-ad427'
    assert get_model_config('/models/Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf')['id']=='llama-unsloth-qwen3-8-flash-next-gguf'
    toolbox=load_toolbox_catalog().toolboxes['intel-b70-llama-sycl-flash-next']
    mtp=get_effective_mtp_config(model,toolbox)
    args=shlex.split(get_mtp_server_args(mtp,'3','1'))
    assert args[args.index('--spec-draft-device')+1]=='SYCL1'
    assert args[args.index('--gpu-layers-draft')+1]=='999'
    assert '--spec-draft-backend-sampling' in args
    assert mtp['draft_models']==['Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf']
    assert model['revision']=='142262902a46f7daed19c79d0771534c8106ad59'
    assert model['auxiliary_downloads'][0]['revision']=='0b2551d191548168d3254ddea4ab943a5ef4f809'
