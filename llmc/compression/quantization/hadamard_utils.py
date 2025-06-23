def apply_exact_had_to_linear(module, had_dim=-1, output=False, R2=None):
    assert isinstance(module, torch.nn.Linear)
    in_features, out_features = module.in_features, module.out_features

    if had_dim != -1:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"

    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    W_ = W_.float().cuda()

    if had_dim == -1:
        if output:
            had_K, K = get_hadK(out_features)
            W_ = matmul_hadU_cuda(W_.t(), had_K, K).t()
        if not output:
            had_K, K = get_hadK(in_features)
            W_ = matmul_hadU_cuda(W_, had_K, K)
    else:
        hadK = hadamard_matrix(had_dim, "cuda").to(torch.float64)
        if R2 is not None:
            hadK = R2.to(torch.float64)
        if output:
            W_ = W_.t()
            transposed_shape = W_.shape
            temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
            temp = temp.to(torch.float64) @ hadK
            W_ = temp.reshape(transposed_shape).t()
        else:
            init_shape = W_.shape
            temp = W_.reshape(-1, init_shape[-1] // had_dim, had_dim)
            temp = temp.to(torch.float64) @ hadK
            W_ = temp.reshape(init_shape)
    module.weight.data = W_.to(device=dev, dtype=dtype)
