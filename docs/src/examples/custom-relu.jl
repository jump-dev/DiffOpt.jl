# # Custom ReLU layer

#md # [![](https://img.shields.io/badge/show-github-579ACA.svg)](@__REPO_ROOT_URL__/docs/src/examples/custom-relu.jl)

# We demonstrate how DiffOpt can be used to generate a simple neural network
# unit - the ReLU layer. A neural network is created using Flux.jl which is
# trained on the MNIST dataset.

# This tutorial uses the following packages

using JuMP
import DiffOpt
import Ipopt
import ChainRulesCore
import Flux
import Statistics
import Base.Iterators: repeated

# ## The ReLU and its derivative

# Define a relu through an optimization problem solved by a quadratic solver.
# Return the solution of the problem.
function matrix_relu(
    y::AbstractArray{T};
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
) where T
    _x = zeros(size(y))
    N = length(y[:, 1])
    empty!(model)
    set_silent(model)
    @variable(model, x[1:N] >= 0)
    for i in 1:size(y, 2)
        @objective(
            model,
            Min,
            x'x -2y[:, i]'x  # x' Q x + q'x with Q = I, q = -2y
        )
        optimize!(model)
        _x[:, i] = value.(x)
    end
    return _x
end


# Define the backward differentiation rule, for the function we defined above.
function ChainRulesCore.rrule(
    ::typeof(matrix_relu),
    y::AbstractArray{T};
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
) where T
    pv = matrix_relu(y, model = model)
    function pullback_matrix_relu(dl_dx)
        ## some value from the backpropagation (e.g., loss) is denoted by `l`
        ## so `dl_dy` is the derivative of `l` wrt `y`
        x = model[:x] # load decision variable `x` into scope
        dl_dy = zeros(T, size(dl_dx))
        dl_dq = zeros(T, size(dl_dx)) # for step-by-step explanation
        for i in 1:size(y, 2)
            MOI.set.(
                model,
                DiffOpt.BackwardInVariablePrimal(),
                x,
                dl_dx[:, i]
            ) # set sensitivities
            DiffOpt.backward(model) # compute grad
            obj_exp = MOI.get(
                model,
                DiffOpt.BackwardOutObjective()
            ) # return gradient wrt objective function parameters
            dl_dq[:, i] = JuMP.coefficient.(obj_exp, x) # coeff of `x` in q'x = -2y'x
            dq_dy = -2 # ∵ dq/dy = -2
            dl_dy[:, i] = dl_dq[:, i] * dq_dy
        end
        return (ChainRulesCore.NoTangent(), dl_dy,)
    end
    return pv, pullback_matrix_relu
end

# For more details about backpropagation, visit [Introduction, ChainRulesCore.jl](https://juliadiff.org/ChainRulesCore.jl/dev/).
# ## prepare data
import MLDatasets
N = 1000
imgs = MLDatasets.MNIST.traintensor(1:N)
labels = MLDatasets.MNIST.trainlabels(1:N);

# Preprocessing
train_X = float.(reshape(imgs, size(imgs, 1) * size(imgs, 2), N)) #stack all the images
train_Y = Flux.onehotbatch(labels, 0:9); # just a common way to encode categorical variables

test_imgs = MLDatasets.MNIST.testtensor(1:N)
test_X = float.(reshape(test_imgs, size(test_imgs, 1) * size(test_imgs, 2), N))
test_Y = Flux.onehotbatch(MLDatasets.MNIST.testlabels(1:N), 0:9);

# ## Define the Network

# Network structure

inner = 15

m = Flux.Chain(
    Flux.Dense(784, inner), #784 being image linear dimension (28 x 28)
    matrix_relu,
    Flux.Dense(inner, 10), # 10 beinf the number of outcomes (0 to 9)
    Flux.softmax,
)

# Define input data
# The original data is repeated `epochs` times because `Flux.train!` only
# loops through the data set once

epochs = 5

dataset = repeated((train_X, train_Y), epochs);

# Parameters for the network training

custom_loss(x, y) = Flux.crossentropy(m(x), y) # training loss function
opt = Flux.ADAM(); # stochastic gradient descent variant to optimize weights of the neral network
evalcb = () -> @show(custom_loss(train_X, train_Y)); # callback to show loss

# Train to optimize network parameters

@time Flux.train!(custom_loss, Flux.params(m), dataset, opt, cb = Flux.throttle(evalcb, 5));

# Although our custom implementation takes time, it is able to reach similar
# accuracy as the usual ReLU function implementation.

accuracy(x, y) = Statistics.mean(Flux.onecold(m(x)) .== Flux.onecold(y)); # average of correct guesses

# Train accuracy

accuracy(train_X, train_Y)

# Test accuracy

accuracy(test_X, test_Y)

# Note that the accuracy is low due to simplified training.
# It is possible to increase the number of samples `N`,
# the number of epochs `epoch` and the connectivity `inner`.
