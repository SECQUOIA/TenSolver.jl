import ITensorMPS
import DynamicPolynomials

# Independent full-space oracle: site 1 is the fastest tensor axis, and the
# sorted spin domain decodes bit b as 2b-1. No tensor-network gate calls here.
qaoa_test_spins(k, n) = [2 * ((k >> (i-1)) & 1) - 1 for i in 1:n]
qaoa_test_vector(psi, sites) = vec(Array(reduce(*, psi), sites...))
function qaoa_test_oracle(Q, l, c, gamma, beta)
  n = length(l)
  energies =
    [dot(s, Q, s) + dot(l, s) + c for s in (qaoa_test_spins(k, n) for k in 0:(2 ^ n - 1))]
  id = Matrix{ComplexF64}(I, 2, 2)
  x = ComplexF64[0 1; 1 0]
  mixer = sum(reduce(kron, [i == j ? x : id for i in n:-1:1]) for j in 1:n)
  uniform = fill(ComplexF64(1 / sqrt(2^n)), 2^n)
  cost = exp(-im * gamma * Diagonal(energies))
  mix = exp(-im * beta * mixer)
  return energies, mix * cost * uniform, cost * mix * uniform
end

function qaoa_test_ray_error(a, b)
  phase = dot(b, a)
  return norm(a - b * phase / abs(phase))
end

@testset "QAOA initialization" begin
  gamma, beta = 0.37, -0.29
  start = TS.QAOAInitialState(gamma, beta)
  constraints = AbstractConstraint[]
  # Predeclared tolerances: norm/Hamiltonian 1e-10, state ray 1e-8,
  # DMRG expectation 1e-8. These fixed angles do not reproduce angle searches.
  fixtures = [
    (
      [0.5, -0.75, 0.25, 1.0],
      [(1, 2, -1.0), (2, 3, 0.75), (3, 4, -0.5), (1, 4, 1.25), (1, 3, -0.25)],
      0.375,
      (-35/8, 49/8),
    ),
    (
      [0.25, -0.5, 0.75, -0.25, 0.5, -0.75],
      [
        (1, 2, 1.0),
        (2, 3, -0.75),
        (3, 4, 0.5),
        (4, 5, -1.25),
        (5, 6, 0.75),
        (1, 6, -0.5),
        (1, 4, 0.25),
        (2, 5, -0.5),
        (3, 6, 1.0),
      ],
      -0.625,
      (-57/8, 47/8),
    ),
    (
      [0.5, -0.25, 0.75, -0.5, 0.25, -0.75, 1.0, -0.25],
      [
        (1, 2, -1.0),
        (2, 3, 0.5),
        (3, 4, -0.75),
        (5, 6, 1.25),
        (6, 7, -0.5),
        (7, 8, 0.75),
        (1, 5, 0.5),
        (2, 6, -1.0),
        (3, 7, 0.75),
        (4, 8, -0.25),
        (1, 8, 1.0),
        (2, 7, -0.5),
      ],
      1.125,
      (-63/8, 81/8),
    ),
  ]

  @testset "Dense oracles: $(length(l)) spins" for (l, edges, offset, exact) in fixtures
    n = length(l)
    Q = zeros(n, n)
    for (i, j, w) in edges
      Q[i, j] = 1.25w
      Q[j, i] = -0.25w
    end
    # Nonzero diagonal and constant must survive domain reduction as an offset.
    Q[1, 1] = 0.625
    c = offset - 0.625
    energies, reference, wrong_order = qaoa_test_oracle(Q, l, c, gamma, beta)
    _, wrong_sign, _ = qaoa_test_oracle(Q, l, c, -gamma, beta)
    @test extrema(energies) == exact
    @test qaoa_test_ray_error(reference, wrong_order) > 1e-3
    @test qaoa_test_ray_error(reference, wrong_sign) > 1e-3

    domain = TS.Domains{Float64}([1, -1], n)
    @test all(==([-1, 1]), domain)
    qr, lr, cr = TS.domain_residue(Q, l, c, domain)
    @test cr == offset
    @test TS.dmrg_initializer(nothing, qr, lr; domain, constraints) ===
          TS.constrained_initial_state
    for preprocess in (false, true)
      qp, lp, cp, dp, cons, p =
        preprocess ? TS.preprocess_model(qr, lr, cr; domain, constraints, cutoff = 1e-12) :
        (qr, lr, cr, domain, constraints, collect(1:n))
      preprocess && @test p != collect(1:n)
      H = TS.tensorize(qp, lp; domain = dp, cutoff = 1e-12)
      sites = ITensorMPS.siteinds(first, H; plev = 0)
      prepare = TS.dmrg_initializer(start, qp, lp; domain = dp, constraints = cons)
      psi = prepare(Float64, sites, ITensorMPS.MPO[]; cutoff = 1e-12, inidim = 16)
      @test ITensorMPS.siteinds(psi) == sites
      @test norm(psi) ≈ 1 atol=1e-10
      @test ITensorMPS.maxlinkdim(psi) <= 16
      v = qaoa_test_vector(psi, sites)
      # Explicitly map tensor-site bits back to original variable positions.
      original_indices =
        [1 + sum(((k >> (j-1)) & 1) << (p[j]-1) for j in 1:n) for k in 0:(2 ^ n - 1)]
      @test qaoa_test_ray_error(v, reference[original_indices]) < 1e-8
      # The dense oracle deliberately materializes up to sixteen physical axes.
      fullH = ITensors.@set_warn_order 18 reduce(*, H)
      denseH = reshape(Array(fullH, ITensors.prime.(sites)..., sites...), 2^n, 2^n)
      @test norm(denseH + cp * I - Diagonal(energies[original_indices])) < 1e-10
      @test real(ITensors.inner(psi', H, psi)) + cp ≈
            real(dot(v, energies[original_indices] .* v)) atol=1e-10
      limited = prepare(Float64, sites, ITensorMPS.MPO[]; cutoff = 1e-12, inidim = 2)
      @test ITensorMPS.maxlinkdim(limited) <= 2
      @test norm(limited) ≈ 1 atol=1e-10
      @test qaoa_test_ray_error(qaoa_test_vector(limited, sites), v) > 1e-4
    end
  end

  @testset "Limiting angles and coefficient cutoff" begin
    Q = [0.0 0.25; 0.125 0.0]
    l = [0.5, -0.75]
    domain = TS.Domains{Float64}([-1, 1], 2)
    sites = ITensorMPS.siteinds(first, TS.tensorize(Q, l; domain, cutoff = 0.0); plev = 0)
    for (g, b) in ((0.0, beta), (gamma, 0.0), (0.0, 0.0))
      psi = TS.qaoa_initial_state(
        TS.QAOAInitialState(g, b),
        Q,
        l,
        sites;
        cutoff = 0.0,
        maxdim = 2,
      )
      _, ref, _ = qaoa_test_oracle(Q, l, 0.0, g, b)
      @test qaoa_test_ray_error(qaoa_test_vector(psi, sites), ref) < 1e-8
    end
    tinyQ = [0.0 1e-13; 0.0 0.0]
    psi =
      TS.qaoa_initial_state(start, tinyQ, [1e-13, 0.0], sites; cutoff = 1e-12, maxdim = 2)
    @test qaoa_test_ray_error(qaoa_test_vector(psi, sites), fill(0.5 + 0im, 4)) < 1e-8
  end

  @testset "Complex MPS through the public solve" begin
    l, edges, c, exact = first(fixtures)
    Q = zeros(4, 4)
    for (i, j, w) in edges
      Q[i, j] = w
    end
    p = TS.qmatrix_permutation(Q; cutoff = 1e-12)
    @test p != collect(1:4)
    energies, _, _ = qaoa_test_oracle(Q[p, p], l[p], c, gamma, beta)
    recorded = Float64[]
    callback = function (psi; iteration, objective, bond_dim, elapsed_time)
      v = qaoa_test_vector(psi, ITensorMPS.siteinds(psi))
      @test eltype(v) <: Complex
      @test objective isa Real
      @test objective ≈ real(dot(v, energies .* v)) atol=1e-8
      push!(recorded, objective)
    end
    # Two trajectories, three sweeps each. No angle search or performance comparison.
    options = (;
      domain = [-1, 1],
      initial_state = start,
      preprocess = true,
      inidim = 16,
      maxdim = 16,
      iterations = 3,
      noise = 0.0,
      cutoff = 1e-12,
      eigsolve_krylovdim = 8,
      eigsolve_maxiter = 5,
      verbosity = 0,
    )
    E, solution = minimize(Q, l, c; options..., on_iteration = callback)
    @test E == exact[1]
    @test recorded == solution.stats.energies
    @test length(recorded) == 3
    @test last(recorded) ≈ exact[1] atol=1e-8
    @test solution.permutation == p
    @test solution.stats.max_bonds.initial_state <= 16
    for _ in 1:16
      s = sample(solution)
      @test all(in((-1, 1)), s)
      @test dot(s, Q, s) + dot(l, s) + c == exact[1]
    end
    Emax, maxsolution = maximize(Q, l, c; options...)
    @test Emax == exact[2]
    # Existing maximize reports stats for its internally negated objective.
    @test last(maxsolution.stats.energies) ≈ -exact[2] atol=1e-8
    s = sample(maxsolution)
    @test dot(s, Q, s) + dot(l, s) + c == exact[2]
  end

  @testset "One site explores both values" begin
    domain = TS.Domains{Float64}([-1, 1], 1)
    H = TS.tensorize(zeros(1, 1), [1.0]; domain, cutoff = 0.0)
    s = only(ITensorMPS.siteinds(first, H; plev = 0))
    psi = ITensorMPS.MPS([ITensors.ITensor(ComplexF64[0, 1], s)])
    E, state = TS.groundstate(H, psi; projections = ITensorMPS.MPO[], cutoff = 1e-12)
    @test E == -1.0
    @test abs(only(state)[s => 1]) ≈ 1.0
    E, solution =
      minimize([1.0], 0.5; domain = [-1, 1], initial_state = start, verbosity = 0)
    @test E == -0.5
    @test sample(solution) == [-1]
  end

  @testset "Unsupported configurations fail clearly" begin
    for angles in ((NaN, 0.0), (0.0, Inf), (big"1e400", 0.0))
      @test_throws ArgumentError TS.QAOAInitialState(angles...)
    end
    for domain in ([0, 1], [-1, 0, 1], [[-1, 1], [0, 1]])
      @test_throws ArgumentError minimize(
        [0.5, -0.25];
        domain,
        initial_state = start,
        verbosity = 0,
      )
    end
    @test_throws ArgumentError minimize(
      [0.5, -0.25];
      domain = [-1, 1],
      initial_state = :qaoa,
    )
    @test_throws ArgumentError minimize(
      [0.5, -0.25];
      domain = [-1, 1],
      initial_state = start,
      constraints = [NotEqualsConstraint([1], [1])],
    )
    @test_throws ArgumentError minimize([NaN, 0.5]; domain = [-1, 1], initial_state = start)
    for inidim in (0, -1, [2, 2])
      @test_throws ArgumentError minimize(
        [0.5, -0.25];
        domain = [-1, 1],
        initial_state = start,
        inidim,
      )
    end
    DynamicPolynomials.@polyvar s[1:2]
    @test_throws ArgumentError minimize(
      s[1] * s[2] + 0.5s[1];
      domain = [-1, 1],
      initial_state = start,
    )
  end
end
