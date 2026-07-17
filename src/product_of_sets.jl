# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    ProductOfSets{T}

This struct is inspired by `MOI.Utilities.@product_of_sets`.

The difference is that the MOI macro requires to know the list of sets at
compile time. In DiffOpt however, the list depends on what the user is going to
use as set as DiffOpt supports any set as long as it implements the required
function of MathOptSetDistances. For this type, the list of sets can be given at
run-time.
"""
mutable struct ProductOfSets{T}
    """
    `rows[i][j]` corresponds to constraint `j` of set type `i`.

    The value depends on `final_touch`:
        * Before `final_touch`, these are `1:dimension` of the constraint
        * After `final_touch`, these are the 1-indexed rows of the full
        constraint matrix
    """
    rows::Vector{Vector{UnitRange{Int}}}

    """
    A sanity bit to check that we don't call functions out-of-order.
    """
    final_touch::Bool

    """
    The set types, and a dictionary mapping S to the integer index. This list is
    defined at run-time.
    """
    set_types::Vector{Type}
    set_types_dict::Dict{Type,Int}

    function ProductOfSets{T}() where {T}
        return new(
            Vector{UnitRange{Int}}[],
            false,
            Type[],
            Dict{Type,Int}(),
        )
    end
end

function MOI.Utilities.set_index(
    set::ProductOfSets,
    ::Type{S},
) where {S<:MOI.AbstractSet}
    return get(set.set_types_dict, S, nothing)
end

MOI.Utilities.set_types(set::ProductOfSets) = set.set_types

function set_set_types(set::ProductOfSets, set_types)
    MOI.empty!(set)
    for S in set_types
        add_set_types(set, S)
    end
    return
end

function add_set_types(set::ProductOfSets, ::Type{S}) where {S}
    if haskey(set.set_types_dict, S)
        return false
    end
    push!(set.rows, Vector{UnitRange{Int}}[])
    push!(set.set_types, S)
    set.set_types_dict[S] = length(set.set_types)
    return true
end

MOI.is_empty(sets::ProductOfSets) = all(isempty, sets.rows)

function MOI.empty!(sets::ProductOfSets)
    map(empty!, sets.rows)
    sets.final_touch = false
    return
end

function MOI.dimension(sets::ProductOfSets)::Int
    @assert sets.final_touch
    for i in reverse(eachindex(sets.rows))
        if !isempty(sets.rows[i])
            return last(sets.rows[i][end])
        end
    end
    return 0  # All rows were empty.
end

function MOI.Utilities.rows(
    sets::ProductOfSets{T},
    ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},S},
)::Int where {T,S}
    @assert sets.final_touch
    i = MOI.Utilities.set_index(sets, S)::Int
    return only(sets.rows[i][ci.value])
end

function MOI.Utilities.rows(
    sets::ProductOfSets{T},
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{T},S},
)::UnitRange{Int} where {T,S}
    @assert sets.final_touch
    i = MOI.Utilities.set_index(sets, S)::Int
    return sets.rows[i][ci.value]
end

function MOI.Utilities.add_set(sets::ProductOfSets, i::Int, dim::Int = 1)::Int64
    @assert !sets.final_touch
    push!(sets.rows[i], 1:dim)
    return length(sets.rows[i])
end

function MOI.Utilities.final_touch(sets::ProductOfSets)::Nothing
    @assert !sets.final_touch
    offset = 0
    for (i, rows) in enumerate(sets.rows)
        for (j, row) in enumerate(rows)
            rows[j] = offset .+ row
            offset += length(row)
        end
    end
    sets.final_touch = true
    return
end

function MOI.Utilities.num_rows(sets::ProductOfSets, ::Type{S})::Int where {S}
    i = MOI.Utilities.set_index(sets, S)::Int
    rows = sets.rows[i]
    if isempty(rows)
        return 0
    elseif sets.final_touch
        return max(0, last(rows[end]) - first(rows[1]) + 1)
    else
        return mapreduce(length, +, rows)
    end
end

function MOI.get(
    sets::ProductOfSets{T},
    ::MOI.ListOfConstraintTypesPresent,
)::Vector{Tuple{Type,Type}} where {T}
    ret = Tuple{Type,Type}[]
    for (i, S) in enumerate(MOI.Utilities.set_types(sets))
        if isempty(sets.rows[i])
            continue
        elseif S <: MOI.AbstractScalarSet
            push!(ret, (MOI.ScalarAffineFunction{T}, S))
        else
            @assert S <: MOI.AbstractVectorSet
            push!(ret, (MOI.VectorAffineFunction{T}, S))
        end
    end
    return ret
end

function MOI.get(
    sets::ProductOfSets,
    ::MOI.NumberOfConstraints{F,S},
)::Int64 where {F,S}
    i = MOI.Utilities.set_index(sets, S)::Union{Nothing,Int}
    if i == nothing
        return 0
    end
    return length(sets.rows[i])
end

function MOI.get(
    sets::ProductOfSets,
    ::MOI.ListOfConstraintIndices{F,S},
)::Vector{MOI.ConstraintIndex{F,S}} where {F,S}
    i = MOI.Utilities.set_index(sets, S)::Union{Nothing,Int}
    if i == nothing
        return MOI.ConstraintIndex{F,S}[]
    end
    return MOI.ConstraintIndex{F,S}.(1:length(sets.rows[i]))
end

function MOI.is_valid(
    sets::ProductOfSets,
    ci::MOI.ConstraintIndex{F,S},
)::Bool where {F,S}
    i = MOI.Utilities.set_index(sets, S)::Union{Nothing,Int}
    if i == nothing
        return false
    end
    return 1 <= ci.value <= length(sets.rows[i])
end
