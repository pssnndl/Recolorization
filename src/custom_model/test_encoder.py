import torch
from encoder import FeatureEncoder
# Test script for the FeatureEncoder model

def test_feature_encoder():
    # Define test input parameters
    batch_size = 4       # Number of images in the batch
    in_channels = 3      # RGB channels
    height, width = 512, 512  # Dimensions of each image

    # Instantiate the FeatureEncoder model
    model = FeatureEncoder(in_channels=in_channels, num_heads=4)

    # Generate synthetic test data (batch of RGB images)
    x = torch.randn(batch_size, in_channels, height, width)  # Shape: (batch_size, in_channels, height, width)

    # Run the model on the test data
    c1, c2, c3, c4 = model(x)

    # Print the shapes of the outputs
    print("Test Data Shapes:")
    print("Input Shape:", x.shape)
    print("c1 (Stage 1 Output) Shape:", c1.shape)  # After DoubleConv + ResNet + SelfAttention + Pooling
    print("c2 (Stage 2 Output) Shape:", c2.shape)  # After DoubleConv + ResNet + SelfAttention + Pooling
    print("c3 (Stage 3 Output) Shape:", c3.shape)  # After DoubleConv + ResNet + SelfAttention + Pooling
    print("c4 (Final Output) Shape:", c4.shape)    # After final DoubleConv without pooling

# Run the test
test_feature_encoder()
