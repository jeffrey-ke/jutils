import torch
import torch.nn as nn

# Adjust imports as needed based on your actual folder structure
from jutils.nn_utils import create_encoder
from jutils.config_classes import EncoderConfig

def test_encoder_with_fatten():
    """
    Tests create_encoder when 'fatten_first' is True.
    Uses vanilla assert statements instead of unittest or pytest.
    """
    config = EncoderConfig(shape=(3, 64, 64), fatten_first=True, fat=32, spatial_reduce_factor=2)
    channel_list = [64, 128, 256]
    encoder = create_encoder(channel_list, config)

    # Check return type
    assert isinstance(encoder, nn.Sequential), "Expected an nn.Sequential from create_encoder."

    # Number of Conv2d layers expected:
    #  1 (fatten) + len(channel_list) = 1 + 3 = 4
    conv_layers_count = sum(1 for layer in encoder if isinstance(layer, nn.Conv2d))
    expected_conv_layers = 1 + len(channel_list)
    assert conv_layers_count == expected_conv_layers, (
        f"Expected {expected_conv_layers} Conv2d layers, got {conv_layers_count}."
    )

    # Check output shape
    x = torch.randn(2, 3, 64, 64)
    output = encoder(x)
    # Check batch size
    assert output.shape[0] == 2, f"Expected batch size of 2, got {output.shape[0]}."
    # Final channel dimension must match last entry in channel_list
    final_channels = channel_list[-1]
    assert output.shape[1] == final_channels, (
        f"Expected output channels = {final_channels}, got {output.shape[1]}."
    )

def test_encoder_without_fatten():
    """
    Tests create_encoder when 'fatten_first' is False.
    """
    config = EncoderConfig(shape=(3, 32, 32), fatten_first=False, spatial_reduce_factor=2)
    channel_list = [32, 64]
    encoder = create_encoder(channel_list, config)

    # Number of Conv2d layers expected = len(channel_list) = 2
    conv_layers_count = sum(1 for layer in encoder if isinstance(layer, nn.Conv2d))
    expected_conv_layers = len(channel_list)
    assert conv_layers_count == expected_conv_layers, (
        f"Expected {expected_conv_layers} Conv2d layers, got {conv_layers_count}."
    )

    # Check output shape
    x = torch.randn(2, 3, 32, 32)
    output = encoder(x)
    assert output.shape[1] == channel_list[-1], (
        f"Expected output channels = {channel_list[-1]}, got {output.shape[1]}."
    )

def main():
    """
    Run all test functions for create_encoder.
    If no assertion fails, it will print a success message.
    """
    test_encoder_with_fatten()
    test_encoder_without_fatten()
    print("All create_encoder tests passed.")

if __name__ == "__main__":
    main()

