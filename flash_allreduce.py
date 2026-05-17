"""
Flash All-Reduce: Algorithm 1 from the paper
Two-step compressed All-Reduce
"""
import torch
import torch.distributed as dist
import time

class FlashAllReduce:
    """
    Implements Algorithm 1 from Flash Communication paper
   
    Standard All-Reduce:
      - Send full precision (32-bit)
      - Communication = bottleneck
   
    Flash All-Reduce:
      - Step 1: Reduce-Scatter with INT4 compression
      - Step 2: All-Gather with INT8 compression
      - 4× less data, 2× faster
    """
   
    def __init__(self, quantizer, world_size, rank):
        self.quantizer = quantizer
        self.world_size = world_size
        self.rank = rank
       
        # Statistics
        self.total_time = 0
        self.num_calls = 0
       
    def all_reduce(self, tensor):
        """
        Main function: Compressed All-Reduce
       
        Args:
            tensor: Tensor to reduce (gradients or activations)
       
        Returns:
            Reduced tensor (sum across all nodes)
        """
        start = time.time()
       
        # Step 1: Split tensor into chunks (one per node)
        chunks = self._split_tensor(tensor)
       
        # Step 2: Reduce-Scatter (compressed)
        reduced_chunk = self._reduce_scatter_quantized(chunks)
       
        # Step 3: All-Gather (compressed)
        final = self._all_gather_quantized(reduced_chunk, tensor.shape)
       
        # Track stats
        self.total_time += time.time() - start
        self.num_calls += 1
       
        return final
   
    def _split_tensor(self, tensor):
        """Split tensor into world_size equal chunks"""
        tensor_flat = tensor.flatten()
        total = tensor_flat.numel()
       
        chunk_size = (total + self.world_size - 1) // self.world_size
       
        # Pad if needed
        if total % self.world_size != 0:
            pad_size = chunk_size * self.world_size - total
            tensor_flat = torch.cat([
                tensor_flat,
                torch.zeros(pad_size, dtype=tensor_flat.dtype, device=tensor_flat.device)
            ])
       
        return tensor_flat.view(self.world_size, chunk_size)
   
    def _reduce_scatter_quantized(self, chunks):
        """
        Step 1: Reduce-Scatter with compression
       
        Process:
        1. Each node quantizes its chunks (INT4)
        2. Send chunk[i] to node i
        3. Each node receives chunks from all others
        4. Dequantize and sum
       
        Result: Each node has 1/world_size of the sum
        """
        # Quantize all chunks
        quant_chunks = []
        for chunk in chunks:
            if hasattr(self.quantizer, 'quantize_for_reduce'):
                qdata = self.quantizer.quantize_for_reduce(chunk)
            else:
                qdata = self.quantizer.quantize(chunk)
            quant_chunks.append(qdata)
       
        # Exchange chunks via All-to-All
        my_idx = self.rank
        received = []
       
        for src in range(self.world_size):
            if src == self.rank:
                received.append(quant_chunks[my_idx])
            else:
                recv = self._exchange_quantized(quant_chunks[src], src)
                received.append(recv)
       
        # Dequantize and sum
        dequant = []
        for qdata in received:
            if hasattr(self.quantizer, 'dequantize_4bit'):
                deq = self.quantizer.dequantize_4bit(qdata)
            else:
                deq = self.quantizer.dequantize(qdata)
            dequant.append(deq)
       
        return torch.stack(dequant).sum(dim=0)
   
    def _all_gather_quantized(self, chunk, original_shape):
        """
        Step 2: All-Gather with compression
       
        Process:
        1. Quantize local chunk (INT8)
        2. Broadcast to all nodes
        3. Dequantize and concatenate
       
        Result: All nodes have complete summed tensor
        """
        # Quantize local chunk
        if hasattr(self.quantizer, 'quantize_for_gather'):
            my_qdata = self.quantizer.quantize_for_gather(chunk)
        else:
            my_qdata = self.quantizer.quantize(chunk)
       
        # Gather from all nodes
        # Every rank calls dist.broadcast for every src in the same order
        # — this is required for collective operations to work correctly
        gathered = []
        for src in range(self.world_size):
            if src == self.rank:
                # Fill buffers with our own data
                recv_quant = my_qdata['quantized'].clone()
                recv_scale = my_qdata['scale'].clone()
                recv_zp    = my_qdata['zero_point'].clone()
            else:
                recv_quant = torch.zeros_like(my_qdata['quantized'])
                recv_scale = torch.zeros_like(my_qdata['scale'])
                recv_zp    = torch.zeros_like(my_qdata['zero_point'])

            # All ranks call broadcast for this src unconditionally — same order on all nodes
            dist.broadcast(recv_quant, src=src)
            dist.broadcast(recv_scale, src=src)
            dist.broadcast(recv_zp,    src=src)

            gathered.append({
                'quantized':      recv_quant,
                'scale':          recv_scale,
                'zero_point':     recv_zp,
                'original_shape': my_qdata['original_shape'],
                'num_elements':   my_qdata['num_elements'],
                'dtype':          my_qdata['dtype'],
                'device':         my_qdata['device']
            })
       
        # Dequantize all
        dequant = []
        for qdata in gathered:
            if hasattr(self.quantizer, 'dequantize_8bit'):
                deq = self.quantizer.dequantize_8bit(qdata)
            else:
                deq = self.quantizer.dequantize(qdata)
            dequant.append(deq)
       
        # Concatenate and reshape
        result = torch.cat(dequant)
        return result[:original_shape.numel()].view(original_shape)
   
    def _exchange_quantized(self, send_data, peer_rank):
        """Send to and receive from peer (blocking)"""
        # Create receive buffers
        recv_quant = torch.zeros_like(send_data['quantized'])
        recv_scale = torch.zeros_like(send_data['scale'])
        recv_zp = torch.zeros_like(send_data['zero_point'])
       
        # Send/receive in order to avoid deadlock
        if self.rank < peer_rank:
            dist.send(send_data['quantized'], dst=peer_rank)
            dist.send(send_data['scale'], dst=peer_rank)
            dist.send(send_data['zero_point'], dst=peer_rank)
            dist.recv(recv_quant, src=peer_rank)
            dist.recv(recv_scale, src=peer_rank)
            dist.recv(recv_zp, src=peer_rank)
        else:
            dist.recv(recv_quant, src=peer_rank)
            dist.recv(recv_scale, src=peer_rank)
            dist.recv(recv_zp, src=peer_rank)
            dist.send(send_data['quantized'], dst=peer_rank)
            dist.send(send_data['scale'], dst=peer_rank)
            dist.send(send_data['zero_point'], dst=peer_rank)
       
        return {
            'quantized': recv_quant,
            'scale': recv_scale,
            'zero_point': recv_zp,
            'original_shape': send_data['original_shape'],
            'num_elements': send_data['num_elements'],
            'dtype': send_data['dtype'],
            'device': send_data['device']
        }
   
    def _broadcast_send(self, data, src):
        """Broadcast data from this node"""
        dist.broadcast(data['quantized'], src=src)
        dist.broadcast(data['scale'], src=src)
        dist.broadcast(data['zero_point'], src=src)
   
    def _broadcast_receive(self, src, template):
        """Receive broadcast from src"""
        recv_quant = torch.zeros_like(template['quantized'])
        recv_scale = torch.zeros_like(template['scale'])
        recv_zp = torch.zeros_like(template['zero_point'])
       
        dist.broadcast(recv_quant, src=src)
        dist.broadcast(recv_scale, src=src)
        dist.broadcast(recv_zp, src=src)
       
        return {
            'quantized': recv_quant,
            'scale': recv_scale,
            'zero_point': recv_zp,
            'original_shape': template['original_shape'],
            'num_elements': template['num_elements'],
            'dtype': template['dtype'],
            'device': template['device']
        }
   
    def get_stats(self):
        """Get performance statistics"""
        avg_time = self.total_time / max(self.num_calls, 1)
        return {
            'num_calls': self.num_calls,
            'total_time': self.total_time,
            'avg_time': avg_time,
            'compression_ratio': self.quantizer.get_compression_ratio()
        }