module MetaMDPs

using Random
using MDPs
import MDPs: state_space, action_space, discount_factor, horizon, action_meaning, state, action, reward, reset!, step!, in_absorbing_state, visualize, factory_reset!

export MetaMDP

mutable struct MetaMDP{S, A} <: AbstractMDP{S, A}
    tasks::Vector{AbstractMDP{S, A}}
    horizon::Int

    task::AbstractMDP{S, A}
    state::S
    action::A
    reward::Float64
    t::Int

    function MetaMDP(tasks, horizon)
        task = tasks[1]
        sspace, aspace = state_space(task), action_space(task)
        S, A = eltype(sspace), eltype(aspace)
        return new{S, A}(tasks, horizon, task,  state(task), action(task), reward(task), 0)
    end
end

function factory_reset!(mm::MetaMDP)
    foreach(mm.tasks, factory_reset!)
    nothing
end


state_space(mm::MetaMDP) = state_space(mm.task)
action_space(mm::MetaMDP) = action_space(mm.task)
action_meaning(mm::MetaMDP{S,A}, a::A) where {S,A} = action_meaning(mm.task, a)
discount_factor(mm::MetaMDP) = 0.99
horizon(mm::MetaMDP) = mm.horizon



state(mm::MetaMDP) = mm.state
action(mm::MetaMDP) = mm.action
reward(mm::MetaMDP) = mm.reward

function reset!(mm::MetaMDP; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing
    mm.task = rand(rng, mm.tasks)
    @debug "Sampled new task" mm.task
    factory_reset!(mm.task)
    reset!(mm.task; rng=rng)
    mm.t = 0
    mm.state = copy(state(mm.task))
    mm.action = action(mm.task)
    mm.reward = reward(mm.task)
    return nothing
end

function step!(mm::MetaMDP{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    @assert a ∈ action_space(mm)
    mm.t += 1
    step!(mm.task, a; rng=rng)
    mm.state = copy(state(mm.task))
    mm.action = a
    mm.reward = reward(mm.task)
    if in_absorbing_state(mm.task)
        reset!(mm.task; rng=rng)
        mm.state = copy(state(mm.task))
    end
    nothing
end


in_absorbing_state(mm::MetaMDP)::Bool =  false  # it's a continual task

visualize(mm::MetaMDP, args...) = visualize(mm.task, args...)

end # module MetaMDPs
