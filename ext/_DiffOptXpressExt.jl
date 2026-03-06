module _DiffOptXpressExt

import DiffOpt
import Xpress
import MathOptInterface as MOI

const BLP = DiffOpt.BasisLinearProgram

BLP._supports_basis_solve(::Xpress.Optimizer) = true

function _get_int_attrib(prob::Xpress.XpressProblem, attr::Int)
    val = Ref{Cint}(0)
    ret = Xpress.Lib.XPRSgetintattrib(prob, attr, val)
    ret != 0 && error("Xpress XPRSgetintattrib failed with return code $ret")
    return Int(val[])
end

"""
    _get_structural_basic_map(s, m)

Return a Dict mapping basis row (1-based) → structural column (1-based) for each
row whose basic variable is a structural (not slack) variable.

Xpress's `XPRSgetpivotorder` returns an array where:
- values `< rows + spare_rows` represent slack/surplus variables
- values `≥ rows + spare_rows` represent structural variables at
  column `value - (rows + spare_rows) + 1` (1-based)
"""
function _get_structural_basic_map(s::Xpress.Optimizer, m::Int)
    ncols = _get_int_attrib(s.inner, Xpress.Lib.XPRS_COLS)
    spare = _get_int_attrib(s.inner, Xpress.Lib.XPRS_SPAREROWS)
    offset = m + spare
    pivot = Vector{Cint}(undef, m)
    ret = Xpress.Lib.XPRSgetpivotorder(s.inner, pivot)
    ret != 0 && error("Xpress XPRSgetpivotorder failed with return code $ret")
    result = Dict{Int,Int}()
    for i in 1:m
        p = Int(pivot[i])
        if offset <= p <= offset + ncols - 1
            result[i] = p - offset + 1  # 1-based column index
        end
    end
    return result
end

function BLP._basis_solve(s::Xpress.Optimizer, db)
    m = _get_int_attrib(s.inner, Xpress.Lib.XPRS_ROWS)

    # Convert CI dict → raw array (ci.value = 1-based row)
    rhs = zeros(m)
    for (ci, val) in db
        rhs[ci.value] = val
    end

    # Solve B * sol = rhs via ftran (in-place)
    if any(!iszero, rhs)
        ret = Xpress.Lib.XPRSftran(s.inner, rhs)
        ret != 0 && error("Xpress ftran failed with return code $ret")
    end

    # Get pivot order to identify basic structural variables
    pivot_map = _get_structural_basic_map(s, m)

    result = Dict{MOI.VariableIndex,Float64}()
    for (row, col_1based) in pivot_map
        result[MOI.VariableIndex(col_1based)] = rhs[row]
    end
    return result
end

function BLP._basis_transpose_solve(
    s::Xpress.Optimizer,
    dx::Dict{MOI.VariableIndex,Float64},
)
    m = _get_int_attrib(s.inner, Xpress.Lib.XPRS_ROWS)

    # Get pivot order to find basis position (row) for each structural variable
    pivot_map = _get_structural_basic_map(s, m)
    col_to_row = Dict{Int,Int}()
    for (row, col_1based) in pivot_map
        col_to_row[col_1based] = row
    end

    # Convert VI dict → raw array indexed by basis position (row)
    rhs = zeros(m)
    for (vi, val) in dx
        row = get(col_to_row, vi.value, 0)
        if row > 0
            rhs[row] = val
        end
    end

    # Solve B' * sol = rhs via btran (in-place)
    if any(!iszero, rhs)
        ret = Xpress.Lib.XPRSbtran(s.inner, rhs)
        ret != 0 && error("Xpress btran failed with return code $ret")
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
            result[row_to_ci[row_1based]] = rhs[row_1based]
        end
    end
    return result
end

end # module
