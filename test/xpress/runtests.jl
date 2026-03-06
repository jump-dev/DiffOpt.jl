# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# ============================================================================
# Xpress community license setup
# ============================================================================

import Xpress_jll
using Downloads

"""
    _setup_xpress_license()

Download and install the Xpress community license if not already present.
The community license is extracted from the `xpress` conda package on
the `fico-xpress` Anaconda channel.
"""
function _setup_xpress_license()
    # Check if already licensed
    try
        Xpress.Optimizer()
        @info "Xpress license already available"
        return
    catch
    end

    # Determine platform and conda package URL
    conda_urls = if Sys.islinux() && Sys.ARCH === :x86_64
        [
            "https://api.anaconda.org/download/fico-xpress/xpress/9.5.7/linux-64/xpress-9.5.7-py310h1d884d0_1772131541.tar.bz2",
        ]
    elseif Sys.isapple() && Sys.ARCH === :aarch64
        [
            "https://api.anaconda.org/download/fico-xpress/xpress/9.5.7/osx-arm64/xpress-9.5.7-py310h50397dc_1772144394.tar.bz2",
        ]
    elseif Sys.isapple() && Sys.ARCH === :x86_64
        [
            "https://api.anaconda.org/download/fico-xpress/xpress/9.5.7/osx-64/xpress-9.5.7-py310hee26cfa_1772144814.tar.bz2",
        ]
    else
        error("Unsupported platform $(Sys.KERNEL)-$(Sys.ARCH) for Xpress community license download")
    end

    # Find artifact directory for license installation
    artifact_dir = Xpress_jll.artifact_dir
    license_dir = joinpath(artifact_dir, "license")
    license_file = joinpath(license_dir, "community-xpauth.xpr")

    if isfile(license_file)
        @info "Community license already installed at $license_file"
        return
    end

    @info "Downloading Xpress community license..."
    tmpdir = mktempdir()
    local pkg_path
    for url in conda_urls
        try
            pkg_path = Downloads.download(url, joinpath(tmpdir, "xpress.tar.bz2"))
            break
        catch e
            @warn "Failed to download from $url: $e"
        end
    end

    # Extract the conda package (tar.bz2)
    extract_dir = joinpath(tmpdir, "extracted")
    mkpath(extract_dir)
    run(`tar xjf $pkg_path -C $extract_dir`)

    # Find the community license file inside the extracted package
    license_src = nothing
    for (root, dirs, files) in walkdir(extract_dir)
        for f in files
            if f == "community-xpauth.xpr"
                license_src = joinpath(root, f)
                break
            end
        end
        license_src !== nothing && break
    end

    if license_src === nothing
        error("Could not find community-xpauth.xpr in downloaded conda package")
    end

    # Install the license
    mkpath(license_dir)
    cp(license_src, license_file; force = true)
    @info "Installed Xpress community license to $license_file"

    # Clean up
    rm(tmpdir; recursive = true, force = true)

    # Verify
    try
        Xpress.Optimizer()
        @info "Xpress license verified successfully"
    catch e
        error("Xpress license installation failed: $e")
    end
    return
end

# ============================================================================
# Test module
# ============================================================================

module TestXpressExtension

include("../BasisLinearProgramTests.jl")

using Test
using JuMP
import DiffOpt
import Xpress
import MathOptInterface as MOI

const ATOL = 1e-6
const RTOL = 1e-6

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function _create_model()
    model = DiffOpt.basis_diff_model(Xpress.Optimizer)
    set_silent(model)
    return model
end

function _create_general_model()
    inner = DiffOpt.diff_optimizer(Xpress.Optimizer)
    MOI.set(
        inner,
        DiffOpt.ModelConstructor(),
        DiffOpt.BasisLinearProgram.GeneralModel,
    )
    model = direct_model(inner)
    set_silent(model)
    return model
end

const BOTH_MODELS = (_create_model, _create_general_model)

# ============================================================================
# Shared tests (forward, reverse, consistency, etc.)
# ============================================================================

function test_common()
    BasisLinearProgramTests.run_common_tests(BOTH_MODELS)
end

# ============================================================================
# Extension loading and auto-detection
# ============================================================================

function test_supports_basis_solve()
    @test DiffOpt.BasisLinearProgram._supports_basis_solve(Xpress.Optimizer())
end

function test_direct_auto_detection()
    model = _create_model()
    @variable(model, x >= 0)
    @variable(model, b in Parameter(5.0))
    @constraint(model, c1, x <= b)
    @objective(model, Min, -x)
    optimize!(model)
    DiffOpt.set_forward_parameter(model, b, 1.0)
    DiffOpt.forward_differentiate!(model)
    bm = backend(model)
    @test MOI.get(bm, DiffOpt.ModelConstructor()) ===
          DiffOpt.BasisLinearProgram.DirectModel
end

function test_allow_direct_false_forces_general_model()
    for create_fn in (_create_general_model,)
        model = create_fn()
        @variable(model, x >= 0)
        @variable(model, b in Parameter(5.0))
        @constraint(model, c1, x <= b)
        @objective(model, Min, -x)
        optimize!(model)
        mc = MOI.get(backend(model), DiffOpt.ModelConstructor())
        @test mc === DiffOpt.BasisLinearProgram.GeneralModel
    end
end

end # module

# Setup license and run tests
_setup_xpress_license()
TestXpressExtension.runtests()
