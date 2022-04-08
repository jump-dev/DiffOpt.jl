# # SVM layer in Flux

#md # [![](https://img.shields.io/badge/show-github-579ACA.svg)](@__REPO_ROOT_URL__/docs/src/examples/svm-flux-layer.jl)


# This tutorial uses the following packages

using JuMP
import DiffOpt
import Ipopt
import ChainRulesCore
import Flux
import Statistics
import Base.Iterators: repeated
using LinearAlgebra

# For more details about backpropagation, visit [Introduction, ChainRulesCore.jl](https://juliadiff.org/ChainRulesCore.jl/dev/).
# ## prepare data
import MLDatasets
N = 1000
imgs = MLDatasets.MNIST.traintensor(1:N)
labels = MLDatasets.MNIST.trainlabels(1:N);

detected_label = 6

# Preprocessing
## stack all the images into D rows, N columns
train_X = float.(reshape(imgs, size(imgs, 1) * size(imgs, 2), N))
train_Y = labels .== detected_label

test_imgs = MLDatasets.MNIST.testtensor(1:N)
test_X = float.(reshape(test_imgs, size(test_imgs, 1) * size(test_imgs, 2), N))
test_Y = MLDatasets.MNIST.testlabels(1:N) .== detected_label

D = size(train_X, 1);

# ## SVM implementation

function SVM(X::AbstractMatrix{T}; model = Model(() -> diff_optimizer(Ipopt.Optimizer))) where {T}
    D, N = size(X)
    
    ## model init
    empty!(model)
    set_optimizer_attribute(model, MOI.Silent(), true)
    
    ## add variables
    @variable(model, l[1:N] >= 0)
    @variable(model, w[1:D])
    @variable(model, b)
    
    @constraint(
        model,
        cons[i in 1:N],
        (2train_Y[i] - 1) * (dot(X[:, i], w) + b) >= 1 - l[i]
    )
        
    @objective(
        model,
        Min,
        sum(l),
    )

    optimize!(model)

    wv = value.(w)
    bv = value(b)
    
    return X'*wv .+ bv
end


# ## Define the classic and DiffOpt-augmented networks

# Network structure

model_standard = Chain(
    Dense(D, 16, Flux.relu),
    Dropout(0.5),
    Dense(16, 32, Flux.relu),
    Dropout(0.5),
    Dense(32, 1, Flux.sigmoid),
)

# Define input data
# The original data is repeated `epochs` times because `Flux.train!` only
# loops through the data set once

epochs = 5

dataset = repeated((train_X, train_Y), epochs);

# Parameters for the network training

# training loss function, Flux optimizer
custom_loss(x, y) = Flux.crossentropy(model_standard(x), y)
opt = Flux.ADAM()
evalcb = () -> @show(custom_loss(train_X, train_Y))

# Train to optimize network parameters

@time Flux.train!(custom_loss, Flux.params(model_standard), dataset, opt, cb = Flux.throttle(evalcb, 5));




# Average of correct guesses
accuracy(x, y) = Statistics.mean(Flux.onecold(m(x)) .== Flux.onecold(y));

# Training accuracy

accuracy(train_X, train_Y)

# Test accuracy

accuracy(test_X, test_Y)

# Note that the accuracy is low due to simplified training.
# It is possible to increase the number of samples `N`,
# the number of epochs `epoch` and the connectivity `inner`.
