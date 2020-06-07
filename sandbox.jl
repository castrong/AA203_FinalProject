using POMDPs
using Random # for AbstractRNG
using POMDPModelTools # for Deterministic
using MCTS # Monte Carlo Tree Search
using POMDPSimulators
using POMDPPolicies
using Colors
using Plots
using LinearAlgebra

# Rectangle for plotting
rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

mutable struct DroneState
    x::Array{Int64, 1}
    y::Array{Int64, 1}
    explored::Array{Bool,2}
end

struct DroneMDP <: MDP{DroneState, Array{Symbol,1}}
    n::Int64
    w::Int64
    h::Int64
    discount::Float64
    terminal_reward::Float64
    start_x::Int64
    start_y::Int64
end
DroneMDP() = DroneMDP(2, 4, 4, 0.95, 100, 1, 1);

POMDPs.discount(m::DroneMDP) = m.discount
POMDPs.isequal(s1::DroneState, s2::DroneState) = (s1.x == s2.x) && (s1.y == s2.y) && (s1.explored == s2.explored)

# For a single state and action
function transition_helper(x, y, w, h, a)
    new_x = x
    new_y = y
    if a == :right
        new_x = x + 1
    elseif a == :left
        new_x = x - 1
    elseif a == :up
        new_y = y + 1
    elseif a == :down
        new_y = y - 1
    else
        println("ERROR UNRECOGNIZED ACTION eee")
    end
    new_x = min(max(new_x, 1), w)
    new_y = min(max(new_y, 1), h)

    return new_x, new_y
end

function POMDPs.gen(m::DroneMDP, s, a, rng)
    # Transition the xs, ys
    new_pairs = transition_helper.(s.x, s.y, m.w, m.h, a)
    new_xs = [pair[1] for pair in new_pairs]
    new_ys = [pair[2] for pair in new_pairs]
    # Make sure all the states we're in are marked as explored
    # also accumulate the reward here
    new_explored = deepcopy(s.explored)
    reward = 0
    for (new_x, new_y) in new_pairs
        if (!new_explored[new_x, new_y])
            reward = reward + 1#/(0.1 + norm([new_x-m.start_x; new_y-m.start_y], 2))
        end
        new_explored[new_x, new_y] = true # easy to extend to all states within radius

    end

    # Create the new state
    new_state = DroneState(new_xs, new_ys, new_explored)
    if (POMDPs.isterminal(m, new_state))
        reward = reward + m.terminal_reward
    end
    return (sp=new_state, r=reward)
end

# Terminate if we've explored all states
POMDPs.isterminal(m::DroneMDP, s::DroneState) = (sum(s.explored)) == m.w * m.h

function idx_to_action(idx)
    actions = [:up :down :left :right]
    return actions[idx+1]
end

function POMDPs.actions(m::DroneMDP)
    return list_of_all_actions
end

## TEST EXAMPLE
n = 5
w = 20
h = 20
terminal_reward = 1000
discount_factor = 0.95
max_steps = 200
start_x = round(Int, w/2);
start_y = round(Int, h/2);

list_of_all_actions = []
for i = 0:4^n-1
    push!(list_of_all_actions, idx_to_action.(digits(i, base = 4, pad = n)))
    # println(list_of_all_actions[end])
end

drone_mdp = DroneMDP(n, w, h, discount_factor, terminal_reward, start_x, start_y)
initial_explored = zeros(Bool, w, h)
initial_explored[start_x, start_y] = true # mark the start location as explored
initial_state = DroneState(start_x * ones(n), start_y * ones(n), initial_explored)
POMDPs.initialstate_distribution(m::DroneMDP) = Deterministic(initial_state)
solver = MCTSSolver(n_iterations=3000, depth=20, exploration_constant=1.0)
planner = solve(solver, drone_mdp)

# simulate
plot([-1],[-1], aspect_ratio=1)
drone_colors = [RGBA(rand(),rand(),rand(),1) for i=1:n]


state_list = []
action_list = []

for (s, a, r) in stepthrough(drone_mdp, planner, "s,a,r", max_steps=max_steps)
    @show s
    @show a
    @show r
    push!(state_list, s)
    push!(action_list, a)

end

# Append terminal state
step_forward = POMDPs.gen(drone_mdp, state_list[end], action_list[end], -1)
next_state = step_forward.sp
if (isterminal(drone_mdp, next_state))
    push!(state_list, next_state)
end

# Loop through and see
anim = @animate for s in state_list
    for i=1:n
        plot!(rectangle(1,1,s.x[i]-1,s.y[i]-1), fillcolor = drone_colors[i],opacity=1,xlims=(0,w),ylims=(0,h),legend=false)
        plot!(s.x[i], s.y[i], )
    end
end
gif(anim, "anim_fps15.gif", fps = 16)


num_left_2 = []
for s in state_list
    push!(num_left, drone_mdp.h * drone_mdp.w - sum(s.explored))
end
plot(num_left, xlabel="Iteration", ylabel="Number Unexplored", title="Unexplored vs. Iteration")



## ANIMATION FUNCTIONS
# using Plots
#
# plot(0:5,0:5,xlims=(0,10),ylims=(0,10))
#
# @gif for i = 1:10
#     plot!(rectangle(i,i,0,0), opacity=.5)
# end
#
# anim = @animate for i = 1:10
#     if i == 1
#         plot(rectangle(i,i,0,0), opacity=.5,xlims=(0,10),ylims=(0,10),legend=false)
#     else
#         plot!(rectangle(i,i,0,0), opacity=.5)
#     end
# end
#
# gif(anim, "anim_fps15.gif", fps = 2)

## SIMPLE POLICY
# policy that maps every input to a feed (true) action
# policy = FunctionPolicy(s->fill(:right, n))


## OLD STUFF
# using Pkg
# Pkg.activate("/Users/cstrong/Desktop/Stanford/FirstYear/SpringQuarter/AA203/AA203_FinalProject")
#
# using POMDPs
# using POMDPToolbox
#
# struct GridWorldState
#     x::Int64 # x position
#     y::Int64 # y position
#     done::Bool # are we in a terminal state?
# end
#
# # initial state constructor
# GridWorldState(x::Int64, y::Int64) = GridWorldState(x,y,false)
# # checks if the position of two states are the same
# posequal(s1::GridWorldState, s2::GridWorldState) = s1.x == s2.x && s1.y == s2.y
#
# # the grid world mdp type
# mutable struct GridWorld <: MDP{GridWorldState, Symbol} # Note that our MDP is parametarized by the state and the action
#     size_x::Int64 # x size of the grid
#     size_y::Int64 # y size of the grid
#     reward_states::Vector{GridWorldState} # the states in which agent recieves reward
#     reward_values::Vector{Float64} # reward values for those states
#     tprob::Float64 # probability of transitioning to the desired state
#     discount_factor::Float64 # discount factor
# end
#
# # we use key worded arguments so we can change any of the values we pass in
# function GridWorld(;sx::Int64=10, # size_x
#                     sy::Int64=10, # size_y
#                     rs::Vector{GridWorldState}=[GridWorldState(4,3), GridWorldState(4,6), GridWorldState(9,3), GridWorldState(8,8)], # reward states
#                     rv::Vector{Float64}=rv = [-10.,-5,10,3], # reward values
#                     tp::Float64=0.7, # tprob
#                     discount_factor::Float64=0.9)
#     return GridWorld(sx, sy, rs, rv, tp, discount_factor)
# end
#
# # we can now create a GridWorld mdp instance like this:
# mdp = GridWorld()
# mdp.reward_states # mdp contains all the defualt values from the constructor
#
# function POMDPs.states(mdp::GridWorld)
#     s = GridWorldState[] # initialize an array of GridWorldStates
#     # loop over all our states, remeber there are two binary variables:
#     # done (d)
#     for d = 0:1, y = 1:mdp.size_y, x = 1:mdp.size_x
#         push!(s, GridWorldState(x,y,d))
#     end
#     return s
# end;
#
# mdp = GridWorld()
# state_space = states(mdp);
# state_space[1]
#
# POMDPs.actions(mdp::GridWorld) = [:up, :down, :left, :right];
#
# # transition helpers
# function inbounds(mdp::GridWorld,x::Int64,y::Int64)
#     if 1 <= x <= mdp.size_x && 1 <= y <= mdp.size_y
#         return true
#     else
#         return false
#     end
# end
#
# inbounds(mdp::GridWorld, state::GridWorldState) = inbounds(mdp, state.x, state.y);
#
# function POMDPs.transition(mdp::GridWorld, state::GridWorldState, action::Symbol)
#     a = action
#     x = state.x
#     y = state.y
#
#     if state.done
#         return SparseCat([GridWorldState(x, y, true)], [1.0])
#     elseif state in mdp.reward_states
#         return SparseCat([GridWorldState(x, y, true)], [1.0])
#     end
#
#     neighbors = [
#         GridWorldState(x+1, y, false), # right
#         GridWorldState(x-1, y, false), # left
#         GridWorldState(x, y-1, false), # down
#         GridWorldState(x, y+1, false), # up
#         ] # See Performance Note below
#
#     targets = Dict(:right=>1, :left=>2, :down=>3, :up=>4) # See Performance Note below
#     target = targets[a]
#
#     probability = fill(0.0, 4)
#
#     if !inbounds(mdp, neighbors[target])
#         # If would transition out of bounds, stay in
#         # same cell with probability 1
#         return SparseCat([GridWorldState(x, y)], [1.0])
#     else
#         probability[target] = mdp.tprob
#
#         oob_count = sum(!inbounds(mdp, n) for n in neighbors) # number of out of bounds neighbors
#
#         new_probability = (1.0 - mdp.tprob)/(3-oob_count)
#
#         for i = 1:4 # do not include neighbor 5
#             if inbounds(mdp, neighbors[i]) && i != target
#                 probability[i] = new_probability
#             end
#         end
#     end
#
#     return SparseCat(neighbors, probability)
# end;
#
#
# function POMDPs.reward(mdp::GridWorld, state::GridWorldState, action::Symbol, statep::GridWorldState) #deleted action
#     if state.done
#         return 0.0
#     end
#     r = 0.0
#     n = length(mdp.reward_states)
#     for i = 1:n
#         if posequal(state, mdp.reward_states[i])
#             r += mdp.reward_values[i]
#         end
#     end
#     return r
# end;
#
# POMDPs.n_states(mdp::GridWorld) = 2*mdp.size_x*mdp.size_y
# POMDPs.n_actions(mdp::GridWorld) = 4
#
# POMDPs.discount(mdp::GridWorld) = mdp.discount_factor;
#
# function POMDPs.stateindex(mdp::GridWorld, state::GridWorldState)
#     sd = Int(state.done + 1)
#     return sub2ind((mdp.size_x, mdp.size_y, 2), state.x, state.y, sd)
# end
# function POMDPs.actionindex(mdp::GridWorld, act::Symbol)
#     if act==:up
#         return 1
#     elseif act==:down
#         return 2
#     elseif act==:left
#         return 3
#     elseif act==:right
#         return 4
#     end
#     error("Invalid GridWorld action: $act")
# end;
#
# POMDPs.isterminal(mdp::GridWorld, s::GridWorldState) = s.done
#
# mdp = GridWorld()
# mdp.tprob=1.0
# sim(mdp, GridWorldState(4,1), max_steps=10) do s
#     println("state is: $s")
#     a = :right
#     println("moving $a")
#     return a
# end;
