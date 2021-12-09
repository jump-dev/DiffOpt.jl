const GeometricConicForm{T} = MOI.Utilities.GenericModel{
    T,
    MOI.Utilities.ObjectiveContainer{Float64},
    MOI.Utilities.VariablesContainer{Float64},
    MOI.Utilities.MatrixOfConstraints{
        Float64,
        MOI.Utilities.MutableSparseMatrixCSC{
            Float64,
            Int,
            # We use `OneBasedIndexing` as it is the same indexing as used
            # by `SparseMatrixCSC` so we can do an allocation-free conversion to
            # `SparseMatrixCSC`.
            MOI.Utilities.OneBasedIndexing,
        },
        Vector{Float64},
        ProductOfSets{Float64},
    },
}


function ConicDiff(model::MOI.ModelLike)
    # For theoretical background, refer Section 3 of Differentiating Through a Cone Program, https://arxiv.org/abs/1904.09043

    vis_src = MOI.get(model, MOI.ListOfVariableIndices())
    cone_types = unique!([S for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())])
    conic_form = GeometricConicForm{Float64}()
    cones = conic_form.constraints.sets
    set_set_types(cones, cone_types)
    index_map = MOI.copy_to(conic_form, model)

    A = -convert(SparseMatrixCSC{Float64, Int}, conic_form.constraints.coefficients)
    b = conic_form.constraints.constants

    if MOI.get(model, MOI.ObjectiveSense()) == MOI.FEASIBILITY_SENSE
        c = spzeros(length(vis_src))
    else
        obj = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
        c = sparse_array_representation(obj, length(vis_src), index_map).terms
        if MOI.get(model, MOI.ObjectiveSense()) == MOI.MAX_SENSE
            c = -c
        end
    end

    # programs in tests were cross-checked against `diffcp`, which follows SCS format
    # hence, some arrays saved during `MOI.optimize!` are not same across all optimizers
    # specifically there's an extra preprocessing step for `PositiveSemidefiniteConeTriangle` constraint for SCS/Mosek

    # get x,y,s
    x = zeros(length(vis_src))
    for vi in vis_src
        i = index_map[vi].value
        x[i] = MOI.get(model, MOI.VariablePrimal(), vi)
    end
    s = map_rows((ci, r) -> MOI.get(model, MOI.ConstraintPrimal(), ci),
        model, cones, index_map, Flattened{Float64}())
    y = map_rows((ci, r) -> MOI.get(model, MOI.ConstraintDual(), ci),
        model, cones, index_map, Flattened{Float64}())

    # pre-compute quantities for the derivative
    m = A.m
    n = A.n
    N = m + n + 1
    # NOTE: w = 1.0 systematically since we asserted the primal-dual pair is optimal
    (u, v, w) = (x, y - s, 1.0)


    # find gradient of projections on dual of the cones
    Dπv = Dπ(v, model, cones, index_map)

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
    vp = π(v, model, cones, index_map)

    return ConicDiff(
        ConicCache(
            M = M,
            vp = vp,
            Dπv = Dπv,
            xys = (x, y, s),
            A = A,
            b = b,
            c = c,
            index_map = index_map,
            cones = cones,
        ),
        nothing,
        nothing,
    )
end
