# Custom ReLU layer

We demonstrate how DiffOpt can be used to generate a simple neural network unit - the ReLU layer. A nueral network is created using Flux.jl which is trained on the MNIST dataset.


```@example 1
using Statistics
using DiffOpt
using Flux
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using OSQP
using JuMP
using ChainRulesCore
```


```@example 1
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

X = X[:, 1:1000]
Y = Y[:, 1:1000];
```


```@example 1
"""
relu method for a Matrix
"""
function matrix_relu(y::AbstractMatrix{T}; model = Model(() -> diff_optimizer(OSQP.Optimizer))) where {T}
    x̂ = zero(y)
    
    # model init
    N = length(y[:, 1])
    empty!(model)
    set_optimizer_attribute(model, MOI.Silent(), true)
    @variable(model, x[1:N] >= zero(T))
    
    for i in 1:size(y)[2]
        @objective(
            model,
            Min,
            x'x -2x'y[:, i]
        )
        optimize!(model)
        x̂[:, i] = value.(x)
    end
    return x̂
end
```




```@example 1
function ChainRulesCore.rrule(::typeof(matrix_relu), y::AbstractArray{T}; model = Model(() -> diff_optimizer(OSQP.Optimizer))) where {T}
    
    pv = matrix_relu(y, model=model) 
    
    function pullback_matrix_relu(dx)
        x = model[:x]
        dy = zero(dx)
        
        for i in 1:size(y)[2]
            MOI.set.(
                model,
                DiffOpt.BackwardInVariablePrimal(),
                x,
                dx[:, i]
            ) 

            DiffOpt.backward(model)  # find grad

            # fetch the objective expression
            obj_exp = MOI.get(model, DiffOpt.BackwardOutObjective())
            
            dy[:, i] = JuMP.coefficient.(obj_exp, x)  # coeff of `x` in -2x'y
            dy[:, i] = -2 * dy[:, i]
        end
        
        return (NO_FIELDS, dy)
    end
    return pv, pullback_matrix_relu
end
```

## Define the Network


```@example 1
m = Chain(
    Dense(784, 64),
    matrix_relu,
    Dense(64, 10),
    softmax,
)
```


```@example 1
custom_loss(x, y) = crossentropy(m(x), y) 
opt = ADAM(); # popular stochastic gradient descent variant

accuracy(x, y) = mean(onecold(m(x)) .== onecold(y)) # cute way to find average of correct guesses

dataset = repeated((X,Y), 20) # repeat the data set, very low accuracy on the orig dataset
evalcb = () -> @show(custom_loss(X, Y)) # callback to show loss
```


Although our custom implementation takes time, it is able to reach similar accuracy as the usual ReLU function implementation.

```@example 1
Flux.train!(custom_loss, params(m), dataset, opt, cb = throttle(evalcb, 5));

@show accuracy(X,Y)
@show accuracy(test_X, test_Y);
```
