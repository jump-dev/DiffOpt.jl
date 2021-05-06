using Statistics
using DiffOpt
using Flux
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using OSQP
using JuMP
using ChainRulesCore

## prepare data
imgs = Flux.Data.MNIST.images()
labels = Flux.Data.MNIST.labels();

# Preprocessing
X = hcat(float.(reshape.(imgs, :))...) #stack all the images
Y = onehotbatch(labels, 0:9); # just a common way to encode categorical variables

test_X = hcat(float.(reshape.(Flux.Data.MNIST.images(:test), :))...)
test_Y = onehotbatch(Flux.Data.MNIST.labels(:test), 0:9)

# float64 to float16, to save memory
X = convert(Array{Float16,2}, X) 
test_X = convert(Array{Float16,2}, test_X)

X = X[:, 1:10000]
Y = Y[:, 1:10000];

"""
relu method for a Matrix
"""
function myRelu(y::AbstractMatrix{T}; model = Model(() -> diff_optimizer(OSQP.Optimizer))) where {T}
    x̂ = zero(y)
    
    # model init
    N = length(y[:, 1])
    empty!(model)
    set_silent(model)
    @variable(model, x[1:N] >= 0)
    @objective(
        model,
        Min,
        x'x -2x'y[:, 1]
    )
    
    for i in 1:size(y, 2)
        set_objective_coefficient.(model, x, -2y[:, i])
        optimize!(model)
        x̂[:, i] = value.(x)
    end
    return x̂
end

function ChainRulesCore.rrule(::typeof(myRelu), y::AbstractArray{T}; model = Model(() -> diff_optimizer(OSQP.Optimizer))) where {T}
    
    pv = myRelu(y, model=model) 
    
    function pullback_myRelu(dx)
        x = model[:x]
        dy = zero(dx)
        
        for i in 1:size(y)[2]
            MOI.set.(
                model,
                DiffOpt.BackwardIn{MOI.VariablePrimal}(), 
                x,
                dx[:, i]
            ) 

            DiffOpt.backward(model)  # find grad

            dy[:, i] = MOI.get.(
                model,
                DiffOpt.BackwardOut{DiffOpt.LinearObjective}(), 
                x, 
            )  # coeff of `x` in -2x'y
            dy[:, i] .= -2 .* dy[:, i]
        end
        
        return (NO_FIELDS, dy)
    end
    return pv, pullback_myRelu
end

m = Chain(
    Dense(784, 64),
    myRelu,
    Dense(64, 10),
    softmax,
)

loss(x, y) = crossentropy(m(x), y) 
opt = ADAM(); # popular stochastic gradient descent variant

accuracy(x, y) = mean(onecold(m(x)) .== onecold(y)) # cute way to find average of correct guesses

dataset = repeated((X,Y), 20) # repeat the data set, very low accuracy on the orig dataset
evalcb = () -> @show(loss(X, Y)) # callback to show loss

Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 5)); #took me ~5 minutes to train on CPU

@show accuracy(X,Y)
@show accuracy(test_X, test_Y);
