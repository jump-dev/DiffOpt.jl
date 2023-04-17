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
        1 -1  1
        1  2  1
        1  0 -1.0
    ]
    a = [1, 2, -3.0]
    b = -5.0
    Q = U' * U
    f = x' * Q * x / 2.0 + a' * x
    c = MOI.add_constraint(bridged, f, MOI.LessThan(-b))
    dQ = [
         1 -1 0
        -1  2 1
         0  1 3.0
    ]
    da = [-1, 1.0, -2]
    db = 3.0
    df = x' * dQ * x / 2.0 + da' * x + db
    MOI.Utilities.final_touch(bridged, nothing)
    MOI.set(bridged, DiffOpt.ForwardConstraintFunction(), c, df)
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
    _test_ΔQ_from_ΔU(U, (d + d') / 2)
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
        2  3.5
        0 -2
    ]
    _test_dU_dQ(U, dU)
    U = [
        1.5 2 -1
        0  -1  3.5
        0   0 -2
    ]
    dU = [
        2.5 -1  2
        0    5 -3
        0    0  3
    ]
    _test_dU_dQ(U, dU)
end

end

TestBridges.runtests()
