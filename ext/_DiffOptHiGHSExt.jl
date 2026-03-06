module _DiffOptHiGHSExt

import DiffOpt
import HiGHS
import MathOptInterface as MOI

const BLP = DiffOpt.BasisLinearProgram

# ============================================================================
# _supports_basis_solve
# ============================================================================

BLP._supports_basis_solve(::HiGHS.Optimizer) = true

# ============================================================================
# MOI-level _basis_solve: Dict{CI, Float64} → Dict{VI, Float64}
# ============================================================================

function BLP._basis_solve(s::HiGHS.Optimizer, db)
    m = Int(HiGHS.Highs_getNumRow(s.inner))

    # Convert CI dict → raw array (at HiGHS level, ci.value = 1-based row)
    rhs = zeros(m)
    for (ci, val) in db
        rhs[ci.value] = val
    end

    # Solve B * sol = rhs
    sol = zeros(m)
    if any(!iszero, rhs)
        ret = HiGHS.Highs_getBasisSolve(s.inner, rhs, sol, C_NULL, C_NULL)
        ret != 0 && error("HiGHS basis_solve failed with return code $ret")
    end

    # Get basic variables and convert result → VI dict
    basic_vars = Vector{Cint}(undef, m)
    ret = HiGHS.Highs_getBasicVariables(s.inner, basic_vars)
    ret != 0 && error("HiGHS getBasicVariables failed with return code $ret")

    result = Dict{MOI.VariableIndex,Float64}()
    for i in 1:m
        bv = Int(basic_vars[i])
        if bv >= 0  # structural variable (0-based column → 1-based VI)
            result[MOI.VariableIndex(bv + 1)] = sol[i]
        end
    end
    return result
end

# ============================================================================
# MOI-level _basis_transpose_solve: Dict{VI, Float64} → Dict{CI, Float64}
# ============================================================================

function BLP._basis_transpose_solve(
    s::HiGHS.Optimizer,
    dx::Dict{MOI.VariableIndex,Float64},
)
    m = Int(HiGHS.Highs_getNumRow(s.inner))

    # Get basic variables to map VIs → basis positions
    basic_vars = Vector{Cint}(undef, m)
    ret = HiGHS.Highs_getBasicVariables(s.inner, basic_vars)
    ret != 0 && error("HiGHS getBasicVariables failed with return code $ret")

    # Convert VI dict → raw array indexed by basis position
    rhs = zeros(m)
    for (vi, val) in dx
        col = vi.value - 1  # 0-based column
        for i in 1:m
            if basic_vars[i] == col
                rhs[i] = val
                break
            end
        end
    end

    # Solve B' * sol = rhs
    sol = zeros(m)
    if any(!iszero, rhs)
        ret = HiGHS.Highs_getBasisTransposeSolve(
            s.inner,
            rhs,
            sol,
            C_NULL,
            C_NULL,
        )
        ret != 0 &&
            error("HiGHS _basis_transpose_solve failed with return code $ret")
    end

    # Convert result → CI dict using MOI queries for row → CI mapping
    row_to_ci = Dict{Int,MOI.ConstraintIndex}()
    for (F, S) in MOI.get(s, MOI.ListOfConstraintTypesPresent())
        F <: MOI.ScalarAffineFunction || continue
        for ci in MOI.get(s, MOI.ListOfConstraintIndices{F,S}())
            row_to_ci[ci.value] = ci
        end
    end

    result = Dict{MOI.ConstraintIndex,Float64}()
    for row_1based in 1:m
        if haskey(row_to_ci, row_1based)
            result[row_to_ci[row_1based]] = sol[row_1based]
        end
    end
    return result
end

end # module
