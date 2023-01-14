module MetaMDPs

using Random
using MDPs
import MDPs: state_space, action_space, action_meaning, state, action, reward, reset!, step!, in_absorbing_state, visualize, factory_reset!

export MetaMDP

mutable struct MetaMDP{S, A} <: AbstractMDP{S, A}
    tasks::Vector{AbstractMDP{S, A}}
    task_horizon::Real
    task_id::Int
    task::AbstractMDP{S, A}
    state::S
    action::A
    reward::Float64
    task_episode_steps::Int

    function MetaMDP(tasks; task_horizon::Real=Inf)
        task = tasks[1]
        sspace, aspace = state_space(task), action_space(task)
        S, A = eltype(sspace), eltype(aspace)
        return new{S, A}(tasks, task_horizon, 0, task,  state(task), action(task), reward(task), 0)
    end
end

function factory_reset!(mm::MetaMDP)
    mm.task_id = 0
    mm.task = mm.tasks[1]
    foreach(mm.tasks, factory_reset!)
    nothing
end


state_space(mm::MetaMDP) = state_space(mm.task)
action_space(mm::MetaMDP) = action_space(mm.task)
action_meaning(mm::MetaMDP{S,A}, a::A) where {S,A} = action_meaning(mm.task, a)


state(mm::MetaMDP) = mm.state
action(mm::MetaMDP) = mm.action
reward(mm::MetaMDP) = mm.reward

function reset!(mm::MetaMDP{S, A}; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    # mm.task = rand(rng, mm.tasks)
    factory_reset!(mm.task)  # factory_reset the outgoing task to free up memory
    mm.task_id = mm.task_id % length(mm.tasks) + 1
    mm.task = mm.tasks[mm.task_id]
    @debug "Sampled new task" mm.task
    factory_reset!(mm.task)
    reset!(mm.task; rng=rng)
    if S == Int
        mm.state = state(mm.task)
    else
        mm.state .= state(mm.task)
    end
    mm.action = action(mm.task)
    mm.reward = reward(mm.task)
    mm.task_episode_steps = 0
    return nothing
end

function step!(mm::MetaMDP{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    @assert a ∈ action_space(mm)
    step!(mm.task, a; rng=rng)
    mm.task_episode_steps += 1
    if S == Int
        mm.state = state(mm.task)
    else
        mm.state .= state(mm.task)
    end
    mm.action = a
    mm.reward = reward(mm.task)
    if in_absorbing_state(mm.task) || mm.task_episode_steps >= mm.task_horizon
        reset!(mm.task; rng=rng)
        mm.task_episode_steps = 0
        if S == Int
            mm.state = state(mm.task)
        else
            mm.state .= state(mm.task)
        end
    end
    nothing
end


in_absorbing_state(mm::MetaMDP)::Bool =  false  # it's a continual task

visualize(mm::MetaMDP, args...; kwargs...) = visualize(mm.task, args...; kwargs...)

end # module MetaMDPs
