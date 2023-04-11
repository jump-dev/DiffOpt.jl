# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    ProductOfSets{T} <: MOI.Utilities.OrderedProductOfSets{T}

The `MOI.Utilities.@product_of_sets` macro requires to know the list of sets
at compile time. In DiffOpt however, the list depends on what the user is going
to use as set as DiffOpt supports any set as long as it implements the
required function of MathOptSetDistances.
For this type, the list of sets can be given a run-time.
"""
mutable struct ProductOfSets{T} <: MOI.Utilities.OrderedProductOfSets{T}
    """
    During the copy, this counts the number of rows corresponding to
    each set. At the end of copy, `final_touch` is called, which
    converts this list into a cumulative ordering.
    """
    num_rows::Vector{Int}

    """
    A dictionary which maps the `set_index` and `offset` of a set to the
    dimension, i.e., `dimension[(set_index,offset)] → dim`.
    """
    dimension::Dict{Tuple{Int,Int},Int}

    """
    A sanity bit to check that we don't call functions out-of-order.
    """
    final_touch::Bool

    set_types::Vector{Type}
    set_types_dict::Dict{Type,Int}

    function ProductOfSets{T}() where {T}
        return new(
            Int[],
            Dict{Tuple{Int,Int},Int}(),
            false,
            Type[],
            Dict{Type,Int}(),
        )
    end
end

function MOI.Utilities.set_index(set::ProductOfSets, S::Type{<:MOI.AbstractSet})
    return get(set.set_types_dict, S, nothing)
end
MOI.Utilities.set_types(set::ProductOfSets) = set.set_types
function set_set_types(set::ProductOfSets, set_types)
    resize!(set.num_rows, length(set_types))
    fill!(set.num_rows, 0)
    resize!(set.set_types, length(set_types))
    copy!(set.set_types, set_types)
    empty!(set.set_types_dict)
    for i in eachindex(set_types)
        set.set_types_dict[set_types[i]] = i
    end
end
function add_set_types(set::ProductOfSets, S::Type)
    if !haskey(set.set_types_dict, S)
        push!(set.num_rows, 0)
        push!(set.set_types, S)
        set.set_types_dict[S] = length(set.set_types)
        return true
    end
    return false
end
