# # Custom ReLU layer

#md # [![](https://img.shields.io/badge/show-github-579ACA.svg)](@__REPO_ROOT_URL__/docs/src/examples/custom-relu.jl)

# We demonstrate how DiffOpt can be used to generate a simple neural network
# unit - the ReLU layer. A neural network is created using Flux.jl and
# trained on the MNIST dataset.

# This tutorial uses the following packages

using JuMP
import DiffOpt
import Ipopt
import ChainRulesCore
import Flux
import MLDatasets
import Statistics
using Base.Iterators: repeated
using LinearAlgebra

# ## The ReLU and its derivative

struct MaxOfN{N}
    w::NTuple{N, Matrix{Float64}}
    b::Vector{Float64}
end

MaxOfN(w::NTuple{N, Matrix{Float64}}) where {N} = MaxOfN{N}(w, randn(N))

Flux.@functor MaxOfN

function (maxofn::MaxOfN)(y::AbstractMatrix; model = direct_model(DiffOpt.diff_optimizer(Ipopt.Optimizer)))
    N, M = size(y)
    empty!(model)
    # set_silent(model)
    @variable(model, x[1:N, 1:M])
    @constraint(model, greater_than_cons[idx in 1:length(maxofn.w)], dot(maxofn.w[idx], x) ≥ maxofn.b[idx])
    @objective(model, Min, dot(x - y, x - y))
    optimize!(model)
    return JuMP.value.(x)
end

# Define the backward differentiation rule, for the function we defined above.
function ChainRulesCore.rrule(maxofn::MaxOfN, y::AbstractMatrix)
    model = direct_model(DiffOpt.diff_optimizer(Ipopt.Optimizer))
    xv = maxofn(y; model = model)
    function pullback_matrix_projection(dl_dx)
        dl_dx = ChainRulesCore.unthunk(dl_dx)
        ##  `dl_dy` is the derivative of `l` wrt `y`
        x = model[:x]
        ## grad wrt input parameters
        dl_dy = zeros(size(dl_dx))
        ## grad wrt layer parameters
        dl_dw = zero.(maxofn.w)
        dl_db = zero(maxofn.b)
        ## set sensitivities
        MOI.set.(model, DiffOpt.BackwardInVariablePrimal(), x, dl_dx)
        ## compute grad
        DiffOpt.backward(model)
        ## compute gradient wrt objective function parameter y
        obj_expr = MOI.get(model, DiffOpt.BackwardOutObjective())
        dl_dy .= -2 * JuMP.coefficient.(obj_expr, x)
        greater_than_cons = model[:greater_than_cons]
        for idx in eachindex(dl_dw)
            cons_expr = MOI.get(model, DiffOpt.BackwardOutConstraint(), greater_than_cons[idx])
            dl_db[idx] = -JuMP.constant(cons_expr)
            dl_dw[idx] .= JuMP.coefficient.(cons_expr, x)
        end
        dself = ChainRulesCore.Tangent{typeof(maxofn)}(; w = dl_dw, b = dl_db)
        return (dself, dl_dy)
    end
    return xv, pullback_matrix_projection
end

# For more details about backpropagation, visit [Introduction, ChainRulesCore.jl](https://juliadiff.org/ChainRulesCore.jl/dev/).
# ## prepare data
N = 1000
imgs = MLDatasets.MNIST.traintensor(1:N)
labels = MLDatasets.MNIST.trainlabels(1:N);

# Preprocessing
train_X = float.(reshape(imgs, size(imgs, 1) * size(imgs, 2), N)) ## stack all the images
train_Y = Flux.onehotbatch(labels, 0:9);

test_imgs = MLDatasets.MNIST.testtensor(1:N)
test_X = float.(reshape(test_imgs, size(test_imgs, 1) * size(test_imgs, 2), N))
test_Y = Flux.onehotbatch(MLDatasets.MNIST.testlabels(1:N), 0:9);

# ## Define the Network

# Network structure

inner = 100

m = Flux.Chain(
    Flux.Dense(784, inner), #784 being image linear dimension (28 x 28)
    MaxOfN((randn(inner,1000), randn(inner,1000), randn(inner,1000))),
    Flux.Dense(inner, 10), # 10 beinf the number of outcomes (0 to 9)
    Flux.softmax,
)

# Define input data
# The original data is repeated `epochs` times because `Flux.train!` only
# loops through the data set once

epochs = 50 # ~1 minute (i7 8th gen with 16gb RAM)
## epochs = 100 # leads to 77.8% in about 2 minutes

dataset = repeated((train_X, train_Y), epochs);

# Parameters for the network training

# training loss function, Flux optimizer
custom_loss(x, y) = Flux.crossentropy(m(x), y)
opt = Flux.ADAM()
evalcb = () -> @show(custom_loss(train_X, train_Y))

# Train to optimize network parameters

@time Flux.train!(custom_loss, Flux.params(m), dataset, opt, cb = Flux.throttle(evalcb, 5));

# Although our custom implementation takes time, it is able to reach similar
# accuracy as the usual ReLU function implementation.

# Average of correct guesses
accuracy(x, y) = Statistics.mean(Flux.onecold(m(x)) .== Flux.onecold(y));

# Training accuracy

accuracy(train_X, train_Y)

# Test accuracy

accuracy(test_X, test_Y)

# Note that the accuracy is low due to simplified training.
# It is possible to increase the number of samples `N`,
# the number of epochs `epoch` and the connectivity `inner`.
