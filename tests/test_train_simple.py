import torch

from train_simple import SimpleTrajectoryModel, fit_fixed_batch


def test_model_shape_and_gradients() -> None:
    model = SimpleTrajectoryModel(hidden_dim=32, residual_blocks=2)
    observation = torch.randn(4, 2, 5)
    target = torch.randn(4, 16, 2)
    prediction = model(observation)

    assert prediction.shape == target.shape
    torch.nn.functional.mse_loss(prediction, target).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_model_rejects_wrong_observation_shape() -> None:
    model = SimpleTrajectoryModel()
    try:
        model(torch.randn(4, 5))
    except ValueError as error:
        assert "Expected observation shape" in str(error)
    else:
        raise AssertionError("Expected the model to reject an invalid input shape.")


def test_overfits_one_fixed_batch() -> None:
    torch.manual_seed(0)
    model = SimpleTrajectoryModel(hidden_dim=32, residual_blocks=1)
    batch = {
        "observation": torch.randn(8, 2, 5),
        "action": torch.randn(8, 16, 2),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    logged = []
    initial, final = fit_fixed_batch(
        model,
        batch,
        optimizer,
        steps=300,
        device=torch.device("cpu"),
        log_loss=lambda step, loss: logged.append((step, loss)),
    )

    assert final < initial * 1e-3
    assert logged[0] == (0, initial)
    assert logged[-1] == (300, final)
