
using Convex, ECOS
Convex.emit_dcp_warnings() = false

function NW(Y::Matrix{Float64}, W::Vector{Float64}, r::Float64, A::Matrix{Float64}, b::Vector{Float64}, task_name::String)
    println("start solving")
    N = size(Y,1)
    dim_z = size(Y,2)

    
    # create decision variables
    if task_name == "shortest_path"
        z = Variable(dim_z, :Bin)
    elseif task_name == "knapsack"
        z = Variable(dim_z)
    end
    z = Variable(dim_z)
    alpha = Variable(1)
    v = Variable(1)
    u = Variable(N)
    beta = Variable(1)

    # create constraints for KL
    cons_v = v > 0
    cons_z0 = z >= 0
    cons_z1 = z <= 1
    cons_u = u >= 0
    cons_array = Array{Constraint}(undef, N)

    Yz = Y*z


    if task_name=="shortest_path"
        for i in 1:N
            cons_array[i] = ExpConstraint((Yz[i]-alpha)*W[i],v,u[i])
        end
        
        cons1 = sum(cons_array) <=  v*exp(-r)*N
        # create constraint Az = b
        cons2 = A*z == b
    elseif task_name=="knapsack"
        for i in 1:N
            cons_array[i] = ExpConstraint((-Yz[i]-alpha)*W[i],v,u[i])
        end
        cons1 = sum(cons_array) <=  v*exp(-r)*N
        # create constraint Az <= b
        cons2 = A*z <= b
    end


    # solve the problem using ECOS solver
    prob = minimize(alpha, cons1, cons2, cons_v, cons_z0, cons_z1, cons_array)

    solve!(prob, ECOS.Optimizer)
    

    x_sol = evaluate(z)
    println("x_sol = @x_sol")
    alpha_sol = evaluate(alpha)

    return x_sol, alpha_sol
end

# create main function
#=
# load data
Y = [1 1;2 2]
W = [1;2]
r = 1.0
A = [1 1]
b = [2]
task_name = "knapsack"

# solve the problem
x_sol, alpha_sol = NW_KL(Y, W, r, A, b, task_name)
=#
