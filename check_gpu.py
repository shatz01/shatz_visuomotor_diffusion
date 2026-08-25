"""Verify that uv, PyTorch, and CUDA can run a small GPU computation."""

import torch


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"Built with CUDA: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch.")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    x = torch.randn(512, 512, device=device)
    result = (x @ x.T).mean()
    torch.cuda.synchronize()
    print(f"GPU computation succeeded: mean={result.item():.6f}")


if __name__ == "__main__":
    main()
