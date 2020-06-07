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
    permissible_map::Array{Bool, 2} # true means permissible
    reward_map::Array{Float64, 2} # reward for exploring each point on map
    potential_map::Array{Float64, 2} # difficulty of reaching each point on the map (ex. altitude)
end
DroneMDP() = DroneMDP(2, 4, 4, 0.95, 100, 1, 1);

POMDPs.discount(m::DroneMDP) = m.discount
POMDPs.isequal(s1::DroneState, s2::DroneState) = (s1.x == s2.x) && (s1.y == s2.y) && (s1.explored == s2.explored)

# For a single state and action
function transition_helper(x, y, a, m)
    w = m.w
    h = m.h
    permissible_map = m.permissible_map

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

    # If it's permissible, update to that. If it's not, return your previous state
    if (permissible_map[new_x, new_y])
        return new_x, new_y
    else
        return x,y
    end
end

function POMDPs.gen(m::DroneMDP, s, a, rng)
    # Transition the xs, ys
    new_pairs = transition_helper.(s.x, s.y, a, [m])
    new_xs = [pair[1] for pair in new_pairs]
    new_ys = [pair[2] for pair in new_pairs]
    # Make sure all the states we're in are marked as explored
    # also accumulate the reward here
    new_explored = deepcopy(s.explored)
    reward = 0
    for (i, new_pair) in enumerate(new_pairs)
        (new_x, new_y) = new_pair
        # add reward if you explore a new location
        if (!new_explored[new_x, new_y])
            reward = reward + m.reward_map[new_x,new_y]
        end
        new_explored[new_x, new_y] = true # easy to extend to all states within radius

        # subtract cost of moving
        reward = reward - max(0,(m.potential_map[new_x, new_y] - m.potential_map[s.x[i], s.y[i]]))
    end

    # Create the new state
    new_state = DroneState(new_xs, new_ys, new_explored)
    if (POMDPs.isterminal(m, new_state))
        reward = reward + m.terminal_reward
    end
    return (sp=new_state, r=reward)
end

# Terminate if we've explored all states
POMDPs.isterminal(m::DroneMDP, s::DroneState) = (sum(s.explored)) == sum(m.permissible_map)

function idx_to_action(idx)
    actions = [:up :down :left :right]
    return actions[idx+1]
end

function POMDPs.actions(m::DroneMDP)
    return list_of_all_actions
end

function add_obstacle(permissible_map, center_x, center_y, radius_x, radius_y)
    permissible_map[center_x-radius_x:center_x+radius_x, center_y-radius_y:center_y+radius_y] .= false
    return permissible_map
end
function add_wall(permissible_map, x_wall, location)
    if (x_wall)
        permissible_map[location, :] .= false
        return permissible_map
    else
        permissible_map[:, location] .= false
        return permissible_map
    end
end

function make_fun_room(w, h, wall_loc)

    permissible_map = fill(true, w, h)
    permissible_map = add_obstacle(permissible_map, 6, 6, 2, 2)
    permissible_map = add_obstacle(permissible_map, 12, 12, 2, 2)

    # permissible_map = fill(true, w, h)
    # permissible_map = add_wall(permissible_map, true, wall_loc)
    # permissible_map = add_wall(permissible_map, false, wall_loc)
    # permissible_map[wall_loc,end-2] = true
    # permissible_map[end-2, wall_loc] = true
    # return permissible_map
end



## TEST EXAMPLE
n = 2
w = 15
h = 10
terminal_reward = 1000
discount_factor = 0.95
max_steps = 30
start_x = round(Int, 6);
start_y = round(Int, 6);

permissible_map = ones(Bool,w,h)#make_fun_room(w, h, 4)
reward_map = ones(w,h)
potential_map = 1*ones(w,h)
potential_map[2:end-1,2:end-1] .= 0
# .05*[20 30 20 50 60 70 90 100 90 60;
#  30 35 25 55 65 75 95 100 90 60;
#  15 20 20 30 40 60 80 90 80 50;
#  05 10 10 20 25 40 50 70 50 40;
#  00 00 05 10 15 20 30 50 30 20;
#  10 20 25 35 30 55 60 70 50 35;
#  20 40 45 60 70 80 90 100 90 100;
#  40 60 70 80 90 90 90 100 90 100;
#  60 80 90 100 100 90 100 80 80 100;
#  70 85 95 100 90 80 80 70 80 60;
#  30 45 50 45 60 50 40 30 20 10;
#  10 15 20 25 35 40 30 20 25 05;
#  00 05 10 15 20 20 10 15 10 00;
#  00 20 25 30 50 70 80 90 100 80;
#  20 30 35 40 60 80 90 90 90 80]

list_of_all_actions = []
for i = 0:4^n-1
    push!(list_of_all_actions, idx_to_action.(digits(i, base = 4, pad = n)))
    # println(list_of_all_actions[end])
end

drone_mdp = DroneMDP(n, w, h, discount_factor, terminal_reward, start_x, start_y, permissible_map, reward_map, potential_map)
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
    println("Gap: ", sum(drone_mdp.permissible_map) - sum(s.explored))
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

# Fill in non-permissible states with black
for i = 1:w
    for j = 1:h
        if !permissible_map[i, j]
            plot!(rectangle(1, 1, i-1, j-1), fillcolor=Colors.RGB(0, 0, 0))
        end
    end
end

anim = @animate for (i, s) in enumerate(state_list)
    old_state = i > 1 ? state_list[i-1] : nothing
    for j=1:n
        # Plot over old drone drawing
        if i > 1
            plot!(rectangle(1,1,old_state.x[j]-1,old_state.y[j]-1), fillcolor = drone_colors[j],opacity=1,xlims=(0,w),ylims=(0,h),legend=false)
        end

        plot!(rectangle(1,1,s.x[j]-1,s.y[j]-1), fillcolor = drone_colors[j],opacity=1,xlims=(0,w),ylims=(0,h),legend=false)
        scatter!(rectangle(0.5, 0.5, s.x[j]-0.75, s.y[j]-0.75),  color="black", opacity=1)


    end
end
gif(anim, "anim_fps15.gif", fps = 4)


# num_left = []
# for s in state_list
#     push!(num_left, drone_mdp.h * drone_mdp.w - sum(s.explored))
# end
# plot(num_left, xlabel="Iteration", ylabel="Number Unexplored", title="Unexplored vs. Iteration")
