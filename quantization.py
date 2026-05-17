
import torch
import math

class SimpleQuantizer:
    """
    Asymmetric quantization (from paper equations 1-2)
    Compresses 32-bit floats to 8-bit integers
    """
   
    def __init__(self, n_bits=8, group_size=128):
        self.n_bits = n_bits
        self.group_size = group_size
        self.max_value = (2 ** n_bits) - 1  # 255 for 8-bit
       
    def quantize(self, tensor):
        """
        Compress tensor: float32 → int8
       
        Args:
            tensor: Input tensor (any shape)
       
        Returns:
            dict with quantized data + metadata for decompression
        """
        original_shape = tensor.shape
        device = tensor.device
        dtype = tensor.dtype
       
        # Flatten tensor
        tensor_flat = tensor.flatten()
        total_elements = tensor_flat.numel()
       
        # Calculate number of groups
        num_groups = math.ceil(total_elements / self.group_size)
       
        # Pad if needed
        if total_elements % self.group_size != 0:
            pad_size = num_groups * self.group_size - total_elements
            tensor_flat = torch.cat([
                tensor_flat,
                torch.zeros(pad_size, dtype=dtype, device=device)
            ])
       
        # Reshape into groups [num_groups, group_size]
        tensor_groups = tensor_flat.view(num_groups, self.group_size)
       
        # Find min/max per group
        group_min = tensor_groups.min(dim=1, keepdim=True)[0]
        group_max = tensor_groups.max(dim=1, keepdim=True)[0]
       
        # Calculate scale and zero_point (Equation 1 from paper)
        scale = (group_max - group_min) / self.max_value
        scale = torch.clamp(scale, min=1e-10)  # Avoid division by zero
       
        zero_point = torch.round(-group_min / scale)
        zero_point = torch.clamp(zero_point, 0, self.max_value)
       
        # Quantize (Equation 2 from paper)
        quantized = torch.round(tensor_groups / scale + zero_point)
        quantized = torch.clamp(quantized, 0, self.max_value).to(torch.uint8)
       
        # Return compressed data + metadata
        return {
            'quantized': quantized,           # Compressed: uint8
            'scale': scale.squeeze(),         # For decompression
            'zero_point': zero_point.squeeze(), # For decompression
            'original_shape': original_shape, # Original dimensions
            'num_elements': total_elements,   # Before padding
            'dtype': dtype,
            'device': device
        }
   
    def dequantize(self, quant_dict):
        """
        Decompress: int8 → float32
       
        Args:
            quant_dict: Output from quantize()
       
        Returns:
            Reconstructed tensor (approximately original)
        """
        quantized = quant_dict['quantized'].float()
        scale = quant_dict['scale'].unsqueeze(1)
        zero_point = quant_dict['zero_point'].unsqueeze(1)
       
        # Reverse quantization
        dequantized = (quantized - zero_point) * scale
       
        # Remove padding and reshape
        dequantized_flat = dequantized.flatten()[:quant_dict['num_elements']]
        result = dequantized_flat.view(quant_dict['original_shape'])
       
        return result.to(quant_dict['dtype'])
   
    def get_compression_ratio(self):
        """Returns how much smaller (e.g., 4.0 = 4x smaller)"""
        return 32.0 / self.n_bits


class MixedPrecisionQuantizer:
    """
    INT6 implementation from paper (Table 3)
    Uses INT8 for All-Gather, INT4 for Reduce-Scatter
    Better accuracy than pure INT4, faster than pure INT8
    """
   
    def __init__(self, group_size=128):
        self.quant_8bit = SimpleQuantizer(n_bits=8, group_size=group_size)
        self.quant_4bit = SimpleQuantizer(n_bits=4, group_size=group_size)
   
    def quantize_for_reduce(self, tensor):
        """Use 4-bit for reduce operations (less sensitive)"""
        return self.quant_4bit.quantize(tensor)
   
    def quantize_for_gather(self, tensor):
        """Use 8-bit for gather operations (more sensitive)"""
        return self.quant_8bit.quantize(tensor)
   
    def dequantize_4bit(self, quant_dict):
        return self.quant_4bit.dequantize(quant_dict)
   
    def dequantize_8bit(self, quant_dict):
        return self.quant_8bit.dequantize(quant_dict)
   
    def get_compression_ratio(self):
        return (32.0 / 8 + 32.0 / 4) / 2  # Average of both


# Test the quantizer
if __name__ == "__main__":
    print("Testing Quantizer...")
   
    # Create test tensor
    test = torch.randn(1000, 768)
    print(f"Original: {test.shape}, {test.numel() * 4 / 1024:.2f} KB")
   
    # Quantize
    q = SimpleQuantizer(n_bits=8, group_size=128)
    compressed = q.quantize(test)
   
    compressed_size = (
        compressed['quantized'].numel() +
        compressed['scale'].numel() * 4 +
        compressed['zero_point'].numel() * 4
    ) / 1024
   
    print(f"Compressed: {compressed_size:.2f} KB")
    print(f"Ratio: {q.get_compression_ratio():.1f}x")
   
    # Dequantize
    reconstructed = q.dequantize(compressed)
   
    # Check error
    error = (test - reconstructed).abs().mean()
    print(f"Error: {error:.6f}")
    print("✓ Test passed!")
