function build_conic_diff_cache!(model)
    # For theoretical background, refer Section 3 of Differentiating Through a Cone Program, https://arxiv.org/abs/1904.09043

    # fetch matrices from MatrixOptInterface
    cone_types = unique!([S for (F, S) in MOI.get(model.optimizer, MOI.ListOfConstraints())])
    # We use `MatOI.OneBasedIndexing` as it is the same indexing as used by `SparseMatrixCSC`
    # so we can do an allocation-free conversion to `SparseMatrixCSC`.
    conic_form = MatOI.GeometricConicForm{Float64, MatOI.SparseMatrixCSRtoCSC{Float64, Int, MatOI.OneBasedIndexing}, Vector{Float64}}(cone_types)
    index_map = MOI.copy_to(conic_form, model)

    # fix optimization sense
    if MOI.get(model, MOI.ObjectiveSense()) == MOI.MAX_SENSE
        conic_form.sense = MOI.MIN_SENSE
        conic_form.c = -conic_form.c
    end

    A = convert(SparseMatrixCSC{Float64, Int}, conic_form.A)
    b = conic_form.b
    c = conic_form.c

    # programs in tests were cross-checked against `diffcp`, which follows SCS format
    # hence, some arrays saved during `MOI.optimize!` are not same across all optimizers
    # specifically there's an extra preprocessing step for `PositiveSemidefiniteConeTriangle` constraint for SCS/Mosek

    # get x,y,s
    x = model.primal_optimal
    s = map_rows((ci, r) -> MOI.get(model, MOI.ConstraintPrimal(), ci), model.optimizer, conic_form, index_map, Flattened{Float64}())
    y = map_rows((ci, r) -> MOI.get(model, MOI.ConstraintDual(), ci), model.optimizer, conic_form, index_map, Flattened{Float64}())

    # pre-compute quantities for the derivative
    m = A.m
    n = A.n
    N = m + n + 1
    # NOTE: w = 1.0 systematically since we asserted the primal-dual pair is optimal
    (u, v, w) = (x, y - s, 1.0)


    # find gradient of projections on dual of the cones
    Dπv = Dπ(v, model.optimizer, conic_form, index_map)

    # Q = [
    #      0   A'   c;
    #     -A   0    b;
    #     -c' -b'   0;
    # ]
    # M = (Q- I) * B + I
    # with B =
    # [
    #  I    .   .    # Πx = x because x is a solution and hence satistfies the constraints
    #  .  Dπv   .
    #  .    .   1    # w >= 0, but in the solution x = 1
    # ]
    # see: https://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf
    # for the definition of Π and why we get I and 1 for x and w respectectively
    # K is defined in (5), Π in sect 2, and projection sin sect 3

    M = [
        spzeros(n,n)     (A' * Dπv)    c
        -A               -Dπv + I      b
        -c'              -b' * Dπv     0.0
    ]
    # find projections on dual of the cones
    vp = π(v, model.optimizer, conic_form, index_map)

    model.gradient_cache =  ConicCache(
        M = M,
        vp = vp,
        Dπv = Dπv,
        xys = (x, y, s),
        A = A,
        b = b,
        c = c,
        index_map = index_map,
        conic_form = conic_form,
    )
    return nothing
end
