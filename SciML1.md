# SciML for quantum chemistry

The simplest quantum chemistry method is Hartree Fock (HF) method, which usually come up with a self consistent field iteration for finding proper single particle wavefunction. Here, we do things a little different. Instead of wavefunction, we formulate the variational method with respect to one body density of a quantum system, and HF method in this case becomes:

$$
\begin{gather}
\min_{\gamma,\boldsymbol{p}}\,\text{tr}\left[\left(\hat{T}+\hat{V}_{\text{ext}}\right)\gamma\right]+V_{ee}[\gamma]\\
\text{s.t.}\left\{\begin{array}{c}
\gamma^{\dagger}=\gamma\\
\gamma^2=\boldsymbol{1}\\
\text{tr}(\gamma)=N
\end{array}\right.
\end{gather}
$$

where $\gamma$ is the one body density matrix and $\boldsymbol{p}$ contains all the variational parameters inside the Hamiltonian.

Computation in quantum chemistry often start with the Hamiltonian
$$\hat{H}=\sum_{p,q}t_{pq}\left(\sum_{\sigma}\hat{c}^{\dagger}_{p\sigma}\hat{c}_{q\sigma}\right)+\frac{1}{2}\sum_{pqrs}\left(\sum_{\sigma\tau}v_{pqrs}\hat{c}^{\dagger}_{p\sigma}\hat{c}^{\dagger}_{r\tau}\hat{c}_{s\tau}\hat{c}_{q\sigma}\right)$$

where $t_{pq}$ contains both the kinetic energy and external potential constribution, and $v_{pqrs}$ is the effect if Coulomb interaction.

Given a set of complete orthonormal basis functions $\left\{\varphi_i\right\}_{i=1}^{n}$, we can explicitly write down the matrix/tensor element of $t_{pq}$ and $v_{pqrs}$ as
$$\begin{gather}
t_{pq}=\int{\varphi_i}^*(x)\left(-\frac{1}{2}\nabla^2-\sum_{I=1}^M\frac{Z}{|r-R_I|}\right)\varphi_j(x)\mathbf{d}x\\
v_{pqrs}=\int{\varphi_p}^*(x){\varphi_r}^*(x^{\prime})\frac{1}{|x-x^{\prime}|}\varphi_s(x^{\prime})\varphi_q(x)\mathbf{d}x\mathbf{d}x^{\prime}
\end{gather}$$
where $M$ equals to the number of atoms inside a molecule.

In quantum chemistry, one commonly used basis function is Gaussian related type of orbits (GTO):
$$\psi^{\text{GTO}}_{ijk}(\vec{r}; \alpha, \vec{R})=N_{ijk}(\alpha)(x-R_x)^i(y-R_y)^j(z-R_z)^k\,e^{-\alpha|\vec{r}-\vec{R}|^2}$$

where the orbit's range is controlled by $\alpha$, and it's centre is located at $\vec{R}$, both of these variables are regarded as variational parameters.

Let's take hydrogen molecule for example, we'll put one $1$s atomic orbital (AO) per each hydrogen atom, and AO is represented as superposition of several GTOs, e.g., in STO-$3$G setting, $1$s AO looks like:
$$\psi^{\text{AO}}(\vec{x};\left\{\alpha_i\right\}_{i=1}^3,\vec{R})=\sum_{i=1}^3C_i\psi^{\text{GTO}}_{000}(\vec{r};\alpha_i,\vec{R})$$
note, $\left\{C_i\right\}_{i=1}^3$ are regarded as fixed for specific AO type.


```julia
using Functors
using BenchmarkTools
using LinearAlgebra
using Random
using Flux
using Zygote
using Optim
using Printf
using Plots
```


```julia
]activate ~/quantumlang
```

    [32m[1m Activating[22m[39m environment at `~/quantumlang/Project.toml`



```julia
using OMEinsum
using QuantumLang
import QuantumLang: get_coords, get_symbols, get_atomic_numbers
```

    ┌ Info: Precompiling QuantumLang [bd262728-6791-47bb-acae-6248b016e32d]
    └ @ Base loading.jl:1278



```julia
# setups
rng = MersenneTwister(123456)
V = Float64
pyplot();
```

    ┌ Info: Precompiling PyPlot [d330b81b-6aea-500a-939a-2ce795aea3ee]
    └ @ Base loading.jl:1278



```julia
# let's setup the configuration of H2 and the basis function in QuantumLang.jl
## build hydrogen molecule
struct H2mole <: Molecule
    r::AbstractParameter
end

@functor H2mole

get_bond_length(h2::H2mole) = value(h2.r)
get_coords(h2::H2mole) = [[0.0, 0.0, 0.0], [0.0, 0.0, get_bond_length(h2)]]
get_symbols(::H2mole) = (:H, :H)
get_atomic_numbers(::H2mole) = (1, 1)

h2 = H2mole(Parameter(1.0, Val(:pos)))      # the bond length is regarded as a parameter

# specify atomic orbital
αs_h = Parameter([3.42525091, 0.62391373, 0.1688554], Val(:pos))   # Parameter is variant
coef_h = Constant([0.15432897, 0.53532814, 0.44463454])            # Constant is invariant
ao_h = AtomicOrbital{0, 0, 0}(αs_h, coef_h)
h_basis = Basis((H=(ao_h,),))

# extract all the variational parameters
ps = params(h2, h_basis);
```

We then use these information to construct operators being used, the `LowdinTransform` is defined as
```julia
function (f::LowdinTransform)(A::Matrix{S}) where {S}
    Ô = MatrixOperator(f.O, (f.ax, f.ax))
    O = Array{S}(Ô)
    Λ, U = eigen(Symmetric(O))
    O_inv_sqrt = U * diagm(one(S) ./ sqrt.(Λ .+ 1e-8)) * U'
    return O_inv_sqrt * A * O_inv_sqrt'
end
```


```julia
# Orbital index passed in to the operator
s1 = MultiIndex((1, 1), (1, 1))     # the first index represents site of hydrogen atom,
s2 = MultiIndex((2, 1), (1, 1))     # the second index represents the number of AO placed at each atom
ax = s1:s2

# build hamiltonian
kin = QuantumLang.Mole.KineticEnergyFunc(h2, h_basis)
T̂ = MatrixOperator(kin, (ax, ax))

coul1e = QuantumLang.Mole.Coulomb1eFunc(h2, h_basis)
V̂ext = MatrixOperator(coul1e, (ax, ax))

coulomb2e = QuantumLang.Mole.Coulomb2eFunc(h2, h_basis)
V̂int = TensorOperator(coulomb2e, (ax, ax, ax, ax))

ρ_mo = DensityMatrix(1, 1)

ovlp = QuantumLang.Mole.OverlapFunc(h2, h_basis)
t_mo2ao = QuantumLang.Mole.LowdinTransform(ovlp, ax);  # Lowdin's transformation is used to orthogonalize the AOs
```

The total energy is obtained from


```julia
function energy()                             # Since we are using real orbitals, we can reduce the computation by exploring symmetry of the matrix
    kin_mat = Array{V}(T̂, Val(:sym))          # T_{pq}=T_{qp}
    Vext_mat = Array{V}(V̂ext, Val(:sym))      # Vext_{pq}=Vext_{qp}
    Vint_tensor = Array{V}(V̂int, Val(:sym))   # Vint_{pqrs}=Vint_{qprs}=Vint_{pqsr}=Vint_{qpsr}=Vint_{rspq}=Vint_{rsqp}=Vint_{srpq}=Vint_{srqp}
    rhf_tensor = Vint_tensor - 0.5*permutedims(Vint_tensor, (1, 3, 4, 2))    # Restricted HF approximation
    ρ_mo_mat = Array{V}(ρ_mo)
    ρ_ao_mat = t_mo2ao(ρ_mo_mat)
    return 2*tr((kin_mat+Vext_mat)*ρ_ao_mat) + 
            2*ein"pqrs,qp,sr->"(rhf_tensor, ρ_ao_mat, ρ_ao_mat)[] +
            QuantumLang.Mole.nuclear_energy(h2)
end

# All parameters are randomly initialized 
let en = energy()
    @show en
end
let grads = gradient(energy, ps)
    @show grads.grads
end
```

    en = 1.1052281523341092
    grads.grads = IdDict{Any,Any}(:(Main.ρ_mo) => (M = [-0.18486491849801423],),:(Main.V̂ext) => (elem_func = (mole = (r = (x = [-6.061357867683221], f = nothing),), basis = (fs = (H = ((αs = (x = [-0.4906984199150126, -3.453901251947399, -1.226359045102295], f = nothing), Cs = (x = nothing,)),),),)), axes = nothing, dims = nothing),:(Main.t_mo2ao) => (O = (mole = (r = (x = [-1.4475915568390392], f = nothing),), basis = (fs = (H = ((αs = (x = [0.05442474525346663, -0.45422233494473196, -0.32399818872825403], f = nothing), Cs = (x = nothing,)),),),)), ax = nothing),:(Main.V̂int) => (elem_func = (mole = (r = (x = [2.2486584135104164], f = nothing),), basis = (fs = (H = ((αs = (x = [-0.05181363121227972, 1.0438732939124071, 0.5108628007302973], f = nothing), Cs = (x = nothing,)),),),)), axes = nothing, dims = nothing),[1.2311747274250842, -0.4717431733756928, -1.7787125516136755] => [0.6232271568873504, 0.8901674781205, -0.2754316387464665],:(Main.T̂) => (elem_func = (mole = (r = (x = [4.002387646086471], f = nothing),), basis = (fs = (H = ((αs = (x = [1.1113144627611762, 3.754417771100224, 0.7640627943537851], f = nothing), Cs = (x = nothing,)),),),)), axes = nothing, dims = nothing),[0.0] => [-2.2579033649253732])





    IdDict{Any,Any} with 7 entries:
      :(Main.ρ_mo)                   => (M = [-0.184865],)
      :(Main.V̂ext)                  => (elem_func = (mole = (r = (x = [-6.06136], …
      :(Main.t_mo2ao)                => (O = (mole = (r = (x = [-1.44759], f = noth…
      :(Main.V̂int)                  => (elem_func = (mole = (r = (x = [2.24866], f…
      [1.23117, -0.471743, -1.77871] => [0.623227, 0.890167, -0.275432]
      :(Main.T̂)                     => (elem_func = (mole = (r = (x = [4.00239], f…
      [0.0]                          => [-2.2579]



We can evaluate the gradient of total energy with respect to all the parameters, most of the gradient evaluation can be handled by `Zygote.jl` except for `Array{V}` function, since it computes the matrix/tensor element and assign them to some preallocated array. Inplace assignment is forbidden in most reverse mode AD framework, but this operation is unavoidable since nearly all of the quantum chemistry algorithms obtain their one and two electron integral by doing element-wise array assignment. To mitigate this problem, We choose to use forward mode differentation method for `Array{V}` function, and manually record it's result on `Zygote`'s `Context`.
```julia
# We define custom adjoint for Array{V} function
@adjoint function Base.Array{T}(op::Union{MatrixOperator, TensorOperator}) where {T}
    N = _num_param_redundant(op)
    op_dual = fmap_dual(op, dual=OffsetDual{N}(0))          # map every optimizable parameter to a dual number
    y, J = extract(Array{Dual{Nothing, T, N}}(op_dual))     # extract the result and jacobian
    return y, function (Δ)
        ∂p = J*vec_scalar(Δ)
        nt, _ = assign_to_named_tuple(__context__, op.elem_func, ∂p)
        ∂op = (elem_func=nt, axes=nothing, dims=nothing)
        return (∂op,)
    end
end
```
> NOTE: since we can regard `Array{V}` function as a map from parameters to matrix/tensor, which is a map of $R^n\to R^{m}$ and $m>>n$, using forward diff method here will bring us performance gain


```julia
# let's benchmark the gradient evaluate process
# for H2 case, 
# `size(kin_mat)=(2, 2)`
# `size(Vext_mat)=(2, 2)`
# `size(Vint_tensor)=(2, 2, 2, 2)`
# num_param = 5

# since we are using mixed mode, we have to allocate extra memory for dual numbers in the forward process, compute and collect the gradient
@benchmark energy()
```




    BenchmarkTools.Trial: 
      memory estimate:  1.14 MiB
      allocs estimate:  11218
      --------------
      minimum time:     1.348 ms (0.00% GC)
      median time:      1.522 ms (0.00% GC)
      mean time:        1.872 ms (8.97% GC)
      maximum time:     21.812 ms (90.76% GC)
      --------------
      samples:          2668
      evals/sample:     1




```julia
@benchmark gradient(energy, ps)
```




    BenchmarkTools.Trial: 
      memory estimate:  2.28 MiB
      allocs estimate:  15512
      --------------
      minimum time:     3.829 ms (0.00% GC)
      median time:      4.405 ms (0.00% GC)
      mean time:        4.877 ms (7.01% GC)
      maximum time:     26.363 ms (68.49% GC)
      --------------
      samples:          1025
      evals/sample:     1


