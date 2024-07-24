import torch


def compute_jacobian(inp: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Computes the derivative of y w.r.t x where y= model(x)

    Args:
        inp: Expected shape: (batch size, total_nodes)
        output: Expected shape: (batch size, total_nodes)

    Returns:
        jacobian: Expected shape: (batch_size, total_nodes, total_nodes) with axis 2 as the argument axis.
        The partial derivative of y[i] w.r.t x is stored along axis 1.
    """
    batch_size = inp.shape[0]
    total_nodes = inp.shape[1]
    device = inp.device

    jacobian_list: list = []
    for node_idx in range(total_nodes):
        # Computing partial derviate: dy[node_idx]/dx
        # Expected shape: (batch_size, total_nodes) as the output is scalar (y[node_idx]) and the input is (total_nodes) dimensional vector
        partial_dev = torch.autograd.grad(
            output[:, node_idx],
            inp,
            grad_outputs=torch.ones(batch_size, device=device),
            retain_graph=True,
            create_graph=True,
        )[0]
        # The shape of gradient is (batch size, total nodes), same as the shape of input src_seq
        jacobian_list.append(partial_dev)
    jacobian = torch.cat(jacobian_list, dim=1).view(batch_size, total_nodes, total_nodes)

    return jacobian
