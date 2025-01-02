mutable struct Experience
    const s::Vector{Float32}  # one-hot encoded if s is discrete
    const a::Vector{Float32}  # one-hot encoded if a is discrete
    r::Float32
    const s′::Vector{Float32}
    terminated::Bool
    truncated::Bool
    const sdim::Int
    const adim::Int
end

function Base.length(e::Experience)
    return e.sdim + e.adim + 1 + e.sdim + 2
end

function get_range(e::Experience, state_lows, state_highs, action_lows, action_highs)
    lows = vcat(state_lows, action_lows, -Inf32, state_lows, 0f0, 0f0) .|> Float32
    highs = vcat(state_highs, action_highs, Inf32, state_highs, 1f0, 1f0) .|> Float32
    return lows, highs
end

function Experience(::UndefInitializer, sdim, adim)
    return Experience(Array{Float32}(undef, sdim), Array{Float32}(undef, adim), 0.0, Array{Float32}(undef, sdim), false, false, sdim, adim)
end

function set_state!(e::Experience, s::AbstractVector{T})::Nothing where T
    e.s .= s
    nothing
end

function set_state!(e::Experience, s::Integer)::Nothing
    fill!(e.s, 0)
    e.s[s] = 1
    nothing
end

function set_action!(e::Experience, a::AbstractVector{T})::Nothing where T
    e.a .= a
    nothing
end

function set_action!(e::Experience, a::Integer)::Nothing
    fill!(e.a, 0)
    e.a[a] = 1
    nothing
end

function set_next_state!(e::Experience, s′::AbstractVector{T})::Nothing where T
    e.s′ .= s′
    nothing
end

function set_next_state!(e::Experience, s′::Integer)::Nothing
    fill!(e.s′, 0)
    e.s′[s′] = 1
    nothing
end

function set_reward!(e::Experience, r::Real)::Nothing
    e.r = r
    nothing
end

function set_terminated!(e::Experience, t::Bool)::Nothing
    e.terminated = t
    nothing
end

function set_truncated!(e::Experience, t::Bool)::Nothing
    e.truncated = t
    nothing
end

function reset_all!(e::Experience)::Nothing
    fill!(e.s, 0)
    fill!(e.a, 0)
    fill!(e.s′, 0)
    e.r = 0.0
    e.terminated = true
    e.truncated = true
    nothing
end

function compile_to_vector!(e::Experience, data::AbstractVector{Float32})
    @assert length(data) == length(e)
    offset = 0
    data[offset+1:offset+e.sdim] .= e.s
    offset += e.sdim
    data[offset+1:offset+e.adim] .= e.a
    offset += e.adim
    data[offset+1] = e.r
    offset += 1
    data[offset+1:offset+e.sdim] .= e.s′
    offset += e.sdim
    data[offset+1] = Float32(e.terminated)
    offset += 1
    data[offset+1] = Float32(e.truncated)
    offset += 1
    return nothing
end




mutable struct MetaMDPObservation
    const current_state::Vector{Float32}  # one-hot encoded if discrete
    latest_experience::Experience
    task_episode_steps::Int
    steps::Int
    const sdim::Int
    const adim::Int
    const max_episode_steps::Int
    const max_steps::Int
    const cat_latest_experience::Bool
    const cat_time::Bool
    const add_time_to_indices::Union{Nothing, Tuple{Int, Int}}
end

MetaMDPObservation(::UndefInitializer, sdim, adim, max_episode_steps, max_steps, cat_latest_experience::Bool, cat_time::Bool, add_time_to_indices) = MetaMDPObservation(Array{Float32}(undef, sdim), Experience(undef, sdim, adim), 0, 0, sdim, adim, max_episode_steps, max_steps, cat_latest_experience, cat_time, add_time_to_indices)

function Base.length(m::MetaMDPObservation)
    return m.sdim + Int(m.cat_latest_experience) * length(m.latest_experience) + Int(m.cat_time) * 2
end

function get_range(m::MetaMDPObservation, state_lows, state_highs, action_lows, action_highs)
    lows = copy(state_lows) .|> Float32
    highs = copy(state_highs) .|> Float32
    if m.cat_latest_experience
        exp_lows, exp_highs = get_range(m.latest_experience, state_lows, state_highs, action_lows, action_highs)
        lows = vcat(lows, exp_lows)
        highs = vcat(highs, exp_highs)
    end
    if m.cat_time
        lows = vcat(lows, zeros(Float32, 2))
        highs = vcat(highs, ones(Float32, 2))
    end
    if m.add_time_to_indices !== nothing
        highs[m.add_time_to_indices[1]] += 1
        highs[m.add_time_to_indices[2]] += 1
    end
    return lows, highs
end

function set_state!(m::MetaMDPObservation, s::AbstractVector{T})::Nothing where T
    m.current_state .= s
    nothing
end

function set_state!(m::MetaMDPObservation, s::Integer)::Nothing
    fill!(m.current_state, 0)
    m.current_state[s] = 1
    nothing
end

function set_steps!(m::MetaMDPObservation, steps::Int)::Nothing
    m.steps = steps
    nothing
end

function set_task_episode_steps!(m::MetaMDPObservation, steps::Int)::Nothing
    m.task_episode_steps = steps
    nothing
end

function reset_all!(m::MetaMDPObservation)::Nothing
    fill!(m.current_state, 0)
    reset_all!(m.latest_experience)
    m.task_episode_steps = 0
    m.steps = 0
    nothing
end

function compile_to_vector!(m::MetaMDPObservation, data::AbstractVector{Float32})
    @assert length(data) == length(m)
    offset = 0
    data[offset+1:offset+m.sdim] .= m.current_state
    offset += m.sdim
    if m.cat_latest_experience
        compile_to_vector!(m.latest_experience, @view data[offset+1:offset+length(m.latest_experience)])
        offset += length(m.latest_experience)
    end
    if m.cat_time
        data[offset+1] = m.task_episode_steps / m.max_episode_steps
        offset += 1
        data[offset+1] = m.steps / m.max_steps
        offset += 1
    end
    if m.add_time_to_indices !== nothing
        data[m.add_time_to_indices[1]] += m.task_episode_steps / m.max_episode_steps
        data[m.add_time_to_indices[2]] += m.steps / m.max_steps
    end
    return nothing
end