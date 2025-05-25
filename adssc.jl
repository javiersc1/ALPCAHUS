using LinearAlgebra
using Optim
using SparseArrays

function adsscWrapper(X::AbstractMatrix, η1::Number, η2::Number; η3::Number=0.0, return_C::Bool=false, K::Int=2)
    A = adssc(X, η1, η2; η3=η3, return_C=return_C)
    A = 0.5*(A + A')
    C = spectralClustering(Matrix(A),K)
    return C
end

function adssc(X::AbstractMatrix, η1::Number, η2::Number; η3::Number=0.0, return_C::Bool=false)
	""" A-DSSC model from 'Doubly Stochastic Subspace Clustering'
	"""
	if η3 < 1e-8 # no l1 regularization
		C = lsr(X, η1; zero_diag=true)
	else
		ensc_gamma = 1/(η1 + η3)
		ensc_tau = η3/(η1 + η3)
		C = ensc(X, ensc_gamma, ensc_tau)
	end
	A = ot_q_dual(abs.(C), η2)

	if return_C
		return A, C
	end

	return A
end

function ot_q_dual(K::Union{AbstractMatrix, SparseMatrixCSC}, γ::Number; verbose::Bool=false, solver::AbstractString="LBFGS")
	""" Quadratically regularized optimal transport
	"""
	n = size(K, 2)
	ν0 = -ones(2*n) .* maximum(K)/2 # initial start
	
	# preallocate space
	col_f = zeros(n)
	col_g = zeros(2*n)

	# objective function
	function f(ν)
		val = 0
		for k in 1:n
			@views col_f .= ν[k+n] .+ ν[1:n] .+ K[:,k]
			clamp!(col_f, 0, Inf)
			col_f .^= 2
			val += sum(col_f)
		end
		val /= 2*γ
		val -= sum(ν)
		return val
	end

	# gradient function
	function g!(G, ν)
		for k in 1:n
			@views col_g[1:n] .= ν[k+n] .+ ν[1:n] .+ K[:,k]
			@views col_g[n+1:end] .= ν[k] .+ ν[n+1:end] .+ K[k,:]
			clamp!(col_g, 0, Inf)
			@views G[k+n] = sum(col_g[1:n])
			@views G[k] = sum(col_g[n+1:end])
		end
		G ./= γ
		G .-= 1
	end

	if solver == "LBFGS"
		results = optimize(f, g!, ν0, LBFGS())
	elseif solver == "GradientDescent"
		results = optimize(f, g!, ν0, GradientDescent())
	else
		throw("Invalid solver")
	end
	ν = Optim.minimizer(results)

	if verbose; println("MinMax α: ", minimum(ν[1:n]), " | ", maximum(ν[1:n])); end
	if verbose; println("MinMax β: ", minimum(ν[n+1:end]), " | ", maximum(ν[n+1:end])); end

	C = spzeros(n,n)
	for i in 1:n
		@views C[:,i] .= clamp.(ν[i+n] .+ ν[1:n] .+ K[:,i], 0, Inf)
	end
	C ./= γ
	return C
end

function lsr(X::AbstractMatrix, γ::Number; zero_diag::Bool=true)
	"""
	'Robust and Efficient Subspace Segmentation via Least Squares Regression'
	Zero diagonal variant.
	"""
	XtX = X'*X
	XtX_γ = XtX + γ * I
	if zero_diag
		D = inv(XtX_γ)
		C = -D * inv(Diagonal(D))
		C .= C - Diagonal(C)
	else
		C = XtX_γ \ XtX
	end
	return C
end
