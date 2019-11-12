package model

import (
	"bitbucket.org/dtolpin/gogp/gp"
	"bitbucket.org/dtolpin/infergo/model"
	. "bitbucket.org/dtolpin/wigp/priors/ad"
	"math"
)

type Model struct {
	Priors *Priors
	GP     *gp.GP
	X      [][]float64
	Y      []float64
	grad   []float64
}

func (m *Model) Observe(x []float64) float64 {
	// Gaussian process
	xGP := make([]float64,
		m.GP.Simil.NTheta()+m.GP.Noise.NTheta()+
			len(m.X)*(m.GP.NDim+1))

	// Kernel parameters
	l := m.GP.Simil.NTheta() + m.GP.Noise.NTheta() // over m.X
	copy(xGP, x[:l])
	k := l
	l += m.Priors.NTheta()

	// Observations
	// ------------
	// Warped inputs
	copy(xGP[k:], m.X[0])
	k += m.GP.NDim
	for i := 0; i != len(m.X)-1; i++ {
		for j := 0; j != m.GP.NDim; j++ {
			xGP[k] = xGP[k-m.GP.NDim] +
				math.Exp(x[l])*(m.X[i+1][j]-m.X[i][j])
			k++
			l++
		}
	}
	// Outputs
	for i := range m.Y {
		xGP[k] = m.Y[i]
		k++
	}

	// Log-likelihood
	// --------------
	llPriors, gPriors := m.Priors.Observe(x), model.Gradient(m.Priors)
	llGP, gGP := m.GP.Observe(xGP), model.Gradient(m.GP)
	ll := llPriors + llGP

	// Gradient
	// --------
	m.grad = make([]float64, len(x))
	copy(m.grad, gPriors)

	// Kernel parameters
	l = 0
	for k = 0; k != m.GP.Simil.NTheta()+m.GP.Noise.NTheta(); k++ {
		m.grad[l] += gGP[k]
		l++
	}
	l += m.Priors.NTheta()

	// Transformations
	for i := 0; i != len(m.X)-1; i++ {
		for j := 0; j != m.GP.NDim; j++ {
			// dLoss/dloglambda = lambda * dx * sum dLoss/dx
			lambda := math.Exp(x[l])
			dx := m.X[i+1][j] - m.X[i][j]
			sum := 0.
			for ii := 1; ii != len(m.X)-i; ii++ {
				sum += gGP[k+ii*m.GP.NDim]
			}
			m.grad[l] += lambda * dx * sum
			k++
			l++
		}
	}

	return ll
}

func (m *Model) Gradient() []float64 {
	return m.grad
}
