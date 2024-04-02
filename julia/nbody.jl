import Pkg
Pkg.add("Threads")
using .Threads

const N = 50000
const STEPS = 3000
const DT = 0.0001
mutable struct Body
    pos::Vector{Float64}
    vel::Vector{Float64}
    mass::Float64
end

function rand_disc()
    theta = rand() * 2 * π
    return [cos(theta) * sqrt(rand()), sin(theta) * sqrt(rand())]
end

function rand_body()
    return Body(rand_disc(), rand_disc(), rand())
end

function update_bodies!(bodies, dt)
    acc = zeros(N, 2)
    d_min = 0.0001

    Threads.@threads for i in 1:N
        acc[i, :] .= 0.0
    end

    Threads.@threads for i in 1:N
        p1 = copy(bodies[i].pos)
        m1 = bodies[i].mass
        @simd for j in (i + 1):N
            p2 = copy(bodies[j].pos)
            m2 = bodies[j].mass

            r = p2 .- p1
            mag_sq = max(dot(r, r), d_min)
            mag = sqrt(mag_sq)
            tmp = r ./ (mag_sq * mag)

            @atomic acc[i, 1] += m2 * tmp[1]
            @atomic acc[i, 2] += m2 * tmp[2]
            @atomic acc[j, 1] -= m1 * tmp[1]
            @atomic acc[j, 2] -= m1 * tmp[2]
        end
    end

    Threads.@threads for i in 1:N
        bodies[i].vel .+= acc[i, :] .* dt
        bodies[i].pos .+= bodies[i].vel .* dt
    end
end

function simulate_and_save(seed, steps, filename)
    srand(seed)
    bodies = [rand_body() for _ in 1:N]

    progress_interval = div(steps, 100)
    start_time = time()

    open(filename, "w") do file
        for i in 1:steps
            for j in 1:N
                println(file, join(bodies[j].pos, ","))
            end
            println(file)

            update_bodies!(bodies, DT)

            if i % progress_interval == 0 || i == steps
                elapsed_time = time() - start_time
                progress = i * 100.0 / steps
                eta = (100.0 - progress) / progress * elapsed_time

                println("Progress: $progress%  ETA: $eta seconds")
            end
        end
    end
end

simulate_and_save(3, STEPS, "simulation_coords.csv")
