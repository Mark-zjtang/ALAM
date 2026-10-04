"""Check shared tokenizer routing and download selection without model execution."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SHARED = 'evaluation/checkpoints/alam/alam_pretrain_latent_action_tokenizer'

def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    server = load('shared_server', 'workflows/pi0_post_training/serve_policy.py')
    train = load('shared_train', 'workflows/pi0_post_training/train.py')
    with patch.dict(os.environ, {}, clear=True):
        for benchmark in ('metaworld', 'libero'):
            args = argparse.Namespace(benchmark=benchmark, suite='spatial', policy_checkpoint=None, alam_checkpoint=None)
            assert server.build_spec(args)['alam_checkpoint'] == SHARED
            assert train.SPECS[benchmark]['alam'] == SHARED
            args.alam_checkpoint = 'custom/tokenizer'
            assert server.build_spec(args)['alam_checkpoint'] == 'custom/tokenizer'
    env = (ROOT / '.env.example').read_text(encoding='utf-8')
    for benchmark in ('METAWORLD', 'LIBERO'):
        assert f'ALAM_{benchmark}_TOKENIZER={SHARED}' in env
    downloader = load('shared_downloader', 'workflows/publishing/download_huggingface.py')
    calls = []
    fake_hub = types.ModuleType('huggingface_hub')
    fake_hub.snapshot_download = lambda **kwargs: calls.append(kwargs)
    with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'huggingface_hub': fake_hub}), patch.object(sys, 'argv', ['download', '--artifact', 'alam_pretrain']), patch.object(downloader, 'repo_path', lambda path: Path(tmp) / path):
        downloader.main()
    assert len(calls) == 1
    assert set(calls[0]['allow_patterns']) == {
        'alam_pretrain_latent_action_tokenizer/config.yaml',
        'alam_pretrain_latent_action_tokenizer/pytorch_model.bin',
    }
    assert calls[0]['repo_id'] == 'Mark-ZJTang/alam_pretrain'
    print('Shared tokenizer routing, CLI override, and download selection: PASS')

if __name__ == '__main__':
    main()
