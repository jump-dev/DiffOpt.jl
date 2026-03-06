module _DiffOptGurobiExt

import DiffOpt
import Gurobi
import MathOptInterface as MOI

const BLP = DiffOpt.BasisLinearProgram

BLP._supports_basis_solve(::Gurobi.Optimizer) = true

function _get_intattr(model, attr::String)
    p = Ref{Cint}()
    ret = Gurobi.Lib.GRBgetintattr(model, attr, p)
    ret != 0 &&
        error("Gurobi GRBgetintattr($attr) failed with return code $ret")
    return Int(p[])
end

"""
    _get_structural_basic_map(s, m, n)

Return a Dict mapping basis position (1-based) → structural column (1-based)
for each basis row whose basic variable is structural (not slack).

Gurobi's `GRBgetBasisHead` returns an array where:
- values `< n` represent structural variables (0-based column index)
- values `≥ n` represent slack variables for row `value - n`
"""
function _get_structural_basic_map(s::Gurobi.Optimizer, m::Int, n::Int)
    bhead = Vector{Cint}(undef, m)
    ret = Gurobi.Lib.GRBgetBasisHead(s.inner, bhead)
    ret != 0 && error("Gurobi GRBgetBasisHead failed with return code $ret")
    result = Dict{Int,Int}()
    for i in 1:m
        bv = Int(bhead[i])
        if 0 <= bv < n  # structural variable (0-based column)
            result[i] = bv + 1  # 1-based column index
        end
    end
    return result
end

"""
    _ci_to_row(s, ci)

Get the 0-based solver row index for a constraint index.
"""
function _ci_to_row(s::Gurobi.Optimizer, ci)
    return s.affine_constraint_info[ci.value].row
end

"""
    _make_svec(ind, val)

Create a GRBsvec from Julia arrays. Returns the struct and keeps
references to the arrays for GC safety (caller must GC.@preserve).
"""
function _make_svec(ind::Vector{Cint}, val::Vector{Cdouble})
    return Gurobi.Lib.GRBsvec(
        Cint(length(ind)),
        isempty(ind) ? Ptr{Cint}(C_NULL) : pointer(ind),
        isempty(val) ? Ptr{Cdouble}(C_NULL) : pointer(val),
    )
end

function BLP._basis_solve(s::Gurobi.Optimizer, db)
    m = _get_intattr(s.inner, "NumConstrs")
    n = _get_intattr(s.inner, "NumVars")

    # Convert CI dict → sparse input vector (0-based row indexing)
    b_ind = Cint[]
    b_val = Cdouble[]
    for (ci, val) in db
        row = _ci_to_row(s, ci)
        push!(b_ind, Cint(row))
        push!(b_val, val)
    end

    # Pre-allocate output arrays
    x_ind = Vector{Cint}(undef, m)
    x_val = Vector{Cdouble}(undef, m)

    x_nz = Cint(0)
    GC.@preserve b_ind b_val x_ind x_val begin
        b_svec = _make_svec(b_ind, b_val)
        x_svec = Gurobi.Lib.GRBsvec(Cint(m), pointer(x_ind), pointer(x_val))

        b_ref = Ref(b_svec)
        x_ref = Ref(x_svec)

        if !isempty(b_ind)
            ret = Gurobi.Lib.GRBFSolve(s.inner, b_ref, x_ref)
            ret != 0 && error("Gurobi GRBFSolve failed with return code $ret")
            x_nz = x_ref[].len
        end
    end

    # Get basis head to identify structural basic variables
    pivot_map = _get_structural_basic_map(s, m, n)

    # Convert sparse result → Dict{VI, Float64}
    # x_ind[k] is the 0-based basis position, x_val[k] is the value
    result = Dict{MOI.VariableIndex,Float64}()
    for k in 1:Int(x_nz)
        basis_pos = Int(x_ind[k]) + 1  # convert to 1-based
        if haskey(pivot_map, basis_pos)
            col_1based = pivot_map[basis_pos]
            result[MOI.VariableIndex(col_1based)] = x_val[k]
        end
    end
    return result
end

function BLP._basis_transpose_solve(
    s::Gurobi.Optimizer,
    dx::Dict{MOI.VariableIndex,Float64},
)
    m = _get_intattr(s.inner, "NumConstrs")
    n = _get_intattr(s.inner, "NumVars")

    # Get basis head: find basis position for each structural variable
    pivot_map = _get_structural_basic_map(s, m, n)
    col_to_pos = Dict{Int,Int}()
    for (pos, col_1based) in pivot_map
        col_to_pos[col_1based] = pos
    end

    # Convert VI dict → sparse input vector (0-based basis position)
    b_ind = Cint[]
    b_val = Cdouble[]
    for (vi, val) in dx
        pos = get(col_to_pos, vi.value, 0)
        if pos > 0
            push!(b_ind, Cint(pos - 1))  # 0-based
            push!(b_val, val)
        end
    end

    # Pre-allocate output arrays
    x_ind = Vector{Cint}(undef, m)
    x_val = Vector{Cdouble}(undef, m)

    x_nz = Cint(0)
    GC.@preserve b_ind b_val x_ind x_val begin
        b_svec = _make_svec(b_ind, b_val)
        x_svec = Gurobi.Lib.GRBsvec(Cint(m), pointer(x_ind), pointer(x_val))

        b_ref = Ref(b_svec)
        x_ref = Ref(x_svec)

        if !isempty(b_ind)
            ret = Gurobi.Lib.GRBBSolve(s.inner, b_ref, x_ref)
            ret != 0 && error("Gurobi GRBBSolve failed with return code $ret")
            x_nz = x_ref[].len
        end
    end

    # Build row → CI mapping using MOI queries
    row_to_ci = Dict{Int,MOI.ConstraintIndex}()
    for (F, S) in MOI.get(s, MOI.ListOfConstraintTypesPresent())
        F <: MOI.ScalarAffineFunction || continue
        for ci in MOI.get(s, MOI.ListOfConstraintIndices{F,S}())
            row = _ci_to_row(s, ci)
            row_to_ci[row] = ci
        end
    end

    # Convert sparse result → Dict{CI, Float64}
    # x_ind[k] is 0-based row position
    result = Dict{MOI.ConstraintIndex,Float64}()
    for k in 1:Int(x_nz)
        row = Int(x_ind[k])  # 0-based
        if haskey(row_to_ci, row)
            result[row_to_ci[row]] = x_val[k]
        end
    end
    return result
end

end # module
