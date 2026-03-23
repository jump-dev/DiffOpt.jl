module TestBridges

using Test

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
end

import MathOptInterface as MOI
import DiffOpt

function test_quad_to_soc()
    model = DiffOpt.ConicProgram.Model()
    F = MOI.VectorAffineFunction{Float64}
    S = MOI.RotatedSecondOrderCone
    # Adds the set type to the list
    @test MOI.supports_constraint(model, F, S)
    bridged = MOI.Bridges.Constraint.QuadtoSOC{Float64}(model)
    x = MOI.add_variables(bridged, 3)
    # We pick orthogonal rows but this is not the `U` that cholesky is going to take in the bridge
    U = [
        1 -1 1
        1 2 1
        1 0 -1.0
    ]
    a = [1, 2, -3.0]
    b = -5.0
    Q = U' * U
    f = x' * Q * x / 2.0 + a' * x
    c = MOI.add_constraint(bridged, f, MOI.LessThan(-b))
    dQ = [
        1 -1 0
        -1 2 1
        0 1 3.0
    ]
    da = [-1, 1.0, -2]
    db = 3.0
    df = x' * dQ * x / 2.0 + da' * x + db
    MOI.Utilities.final_touch(bridged, nothing)
    return MOI.set(bridged, DiffOpt.ForwardConstraintFunction(), c, df)
end

function _test_dU_from_dQ(U, dU)
    dQ = dU'U + U'dU
    _dU = copy(dQ)
    __dU = copy(dQ)
    # Compiling
    DiffOpt.dU_from_dQ!(__dU, U)
    @test @allocated(DiffOpt.dU_from_dQ!(_dU, U)) == 0
    @test _dU ≈ dU
end

function _test_ΔQ_from_ΔU(U, ΔQ)
    ΔU = U * (ΔQ + ΔQ')
    _ΔQ = copy(ΔU)
    __ΔQ = copy(ΔU)
    # Compiling
    DiffOpt.ΔQ_from_ΔU!(__ΔQ, U)
    @test @allocated(DiffOpt.ΔQ_from_ΔU!(_ΔQ, U)) == 0
    @test _ΔQ ≈ ΔQ
end

function _test_dU_dQ(U, d)
    _test_dU_from_dQ(U, d)
    return _test_ΔQ_from_ΔU(U, (d + d') / 2)
end

function test_dU_from_dQ()
    _test_dU_dQ(2ones(1, 1), 3ones(1, 1))
    U = [
        1 2
        0 1
    ]
    dU = [
        1 -1
        0 1
    ]
    _test_dU_dQ(U, dU)
    U = [
        -3 5
        0 2.5
    ]
    dU = [
        2 3.5
        0 -2
    ]
    _test_dU_dQ(U, dU)
    U = [
        1.5 2 -1
        0 -1 3.5
        0 0 -2
    ]
    dU = [
        2.5 -1 2
        0 5 -3
        0 0 3
    ]
    return _test_dU_dQ(U, dU)
end

function _make_square_bridge(dim)
    s = MOI.PositiveSemidefiniteConeSquare(dim)
    return MOI.Bridges.Constraint.SquareBridge{
        Float64,
        MOI.VectorAffineFunction{Float64},
        MOI.ScalarAffineFunction{Float64},
        MOI.PositiveSemidefiniteConeTriangle,
        MOI.PositiveSemidefiniteConeSquare,
    }(
        s,
        MOI.ConstraintIndex{
            MOI.VectorAffineFunction{Float64},
            MOI.PositiveSemidefiniteConeTriangle,
        }(
            1,
        ),
        Pair{
            Tuple{Int,Int},
            MOI.ConstraintIndex{
                MOI.ScalarAffineFunction{Float64},
                MOI.EqualTo{Float64},
            },
        }[],
    )
end

function test_square_to_triangle_indices()
    # 2x2: square col-major [a11, a21, a12, a22] → upper tri [a11, a12, a22]
    @test DiffOpt._square_to_triangle_indices(_make_square_bridge(2)) ==
          [1, 3, 4]
    # 3x3: square col-major [a11,a21,a31, a12,a22,a32, a13,a23,a33]
    #   → upper tri [a11, a12,a22, a13,a23,a33] at indices [1, 4,5, 7,8,9]
    @test DiffOpt._square_to_triangle_indices(_make_square_bridge(3)) ==
          [1, 4, 5, 7, 8, 9]
    # 1x1: trivial
    @test DiffOpt._square_to_triangle_indices(_make_square_bridge(1)) == [1]
end

function _make_rootdet_square_bridge(dim)
    s = MOI.RootDetConeSquare(dim)
    return MOI.Bridges.Constraint.SquareBridge{
        Float64,
        MOI.VectorAffineFunction{Float64},
        MOI.ScalarAffineFunction{Float64},
        MOI.RootDetConeTriangle,
        MOI.RootDetConeSquare,
    }(
        s,
        MOI.ConstraintIndex{
            MOI.VectorAffineFunction{Float64},
            MOI.RootDetConeTriangle,
        }(
            1,
        ),
        Pair{
            Tuple{Int,Int},
            MOI.ConstraintIndex{
                MOI.ScalarAffineFunction{Float64},
                MOI.EqualTo{Float64},
            },
        }[],
    )
end

function test_square_to_triangle_indices_with_offset()
    # RootDetConeSquare(2) has 1 offset entry (the rootdet variable t)
    # Full square: [t, a11, a21, a12, a22] → triangle: [t, a11, a12, a22]
    bridge = _make_rootdet_square_bridge(2)
    @test DiffOpt._square_to_triangle_indices(bridge) == [1, 2, 4, 5]
end

function test_square_offset()
    @test DiffOpt._square_offset(MOI.PositiveSemidefiniteConeSquare(2)) == 0
    @test DiffOpt._square_offset(MOI.RootDetConeSquare(2)) == 1
    @test DiffOpt._square_offset(MOI.LogDetConeSquare(2)) == 2
    @test MOI.Bridges.Constraint._square_offset(
        MOI.PositiveSemidefiniteConeSquare(2),
    ) == 0
    @test MOI.Bridges.Constraint._square_offset(MOI.RootDetConeSquare(2)) == 1
    @test MOI.Bridges.Constraint._square_offset(MOI.LogDetConeSquare(2)) == 2
end

function test_triangle_to_square_scalars()
    # 2x2 PSD: triangle [a11, a12, a22] → square [a11, a12, a12, a22]
    s = MOI.PositiveSemidefiniteConeSquare(2)
    @test DiffOpt._triangle_to_square_scalars([1, 2, 3], s) == [1, 2, 2, 3]
    # 3x3 PSD: triangle [a11, a12, a22, a13, a23, a33]
    #   → square col-major [a11, a12, a13, a12, a22, a23, a13, a23, a33]
    s3 = MOI.PositiveSemidefiniteConeSquare(3)
    @test DiffOpt._triangle_to_square_scalars([1, 2, 3, 4, 5, 6], s3) ==
          [1, 2, 4, 2, 3, 5, 4, 5, 6]
    # RootDetConeSquare(2): triangle [t, a11, a12, a22]
    #   → square [t, a11, a12, a12, a22]
    sr = MOI.RootDetConeSquare(2)
    @test DiffOpt._triangle_to_square_scalars([10, 1, 2, 3], sr) ==
          [10, 1, 2, 2, 3]
    # LogDetConeSquare(2): triangle [u, t, a11, a12, a22]
    #   → square [u, t, a11, a12, a12, a22]
    sl = MOI.LogDetConeSquare(2)
    @test DiffOpt._triangle_to_square_scalars([10, 20, 1, 2, 3], sl) ==
          [10, 20, 1, 2, 2, 3]
end

end

TestBridges.runtests()
