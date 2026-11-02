"""
Check files written by safetensors.Write (run by TestWriteMatchesPython) with
the reference implementation:

    .venv/bin/python scripts/check_safetensors_write.py <written> <rounded>

<written> must load with safetensors, and be byte for byte what save_file
writes for the same tensors, dtypes and metadata. <rounded> holds the same
values as float32 ("as_f32") and as Go's bf16 ("as_bf16"), which must be what
PyTorch's .to(torch.bfloat16) makes of them.
"""
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: python {sys.argv[0]} <written> <rounded>")
    written, rounded = sys.argv[1:]

    with safe_open(written, "pt") as f:
        metadata = f.metadata()
        tensors = {name: f.get_tensor(name) for name in f.keys()}
    if metadata != {"format": "pt"}:
        sys.exit(f"metadata {metadata}")
    again = written + ".python"
    save_file(tensors, again, metadata=metadata)
    with open(written, "rb") as a, open(again, "rb") as b:
        got, want = a.read(), b.read()
    if got != want:
        n = int.from_bytes(want[:8], "little")
        sys.exit(f"the file differs from save_file's\nGo:     {got[8:8 + n]!r}\nPython: {want[8:8 + n]!r}")

    r = load_file(rounded)
    from_torch = r["as_f32"].to(torch.bfloat16)
    go, py = r["as_bf16"], from_torch
    nans = py.isnan()
    if not torch.equal(go.isnan(), nans):
        sys.exit("NaNs differ")
    same = go.to(torch.float32)[~nans] == py.to(torch.float32)[~nans]
    signs = go.to(torch.float32)[~nans].signbit() == py.to(torch.float32)[~nans].signbit()
    if not (same.all() and signs.all()):
        i = (~(same & signs)).nonzero()[0].item()
        sys.exit(f"value {i}: {r['as_f32'][~nans][i].item()!r} rounds to {go[~nans][i].item()!r}, PyTorch {py[~nans][i].item()!r}")
    print(f"{len(tensors)} tensors load and match save_file byte for byte; {len(go)} values round as PyTorch's")


if __name__ == "__main__":
    main()
