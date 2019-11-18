package priors

import (
	"bitbucket.org/dtolpin/infergo/model"
	. "bitbucket.org/dtolpin/infergo/dist"
	"math"
)

type Priors interface {
	model.Model
	NTheta() int
}

type ARPriors struct {
}

func (m *ARPriors) NTheta() int {
	return 1
}

func (m *ARPriors) Observe(x []float64) float64 {
	const (
		c  = iota // variance
		l         // length scale
		s         // noise variance
		t		  // warping
		i0        // first transformation (relative step change)
	)

	ll := 0.

	// Priors of the Gaussian process
	// ------------------------------
	// Variance is mostly less than 1.
	ll += Normal.Logp(-1, 1, x[c])
	// Length scale is around 1, in wide margins.
	ll += Normal.Logp(-2, 2, x[l])
	// Noise variance is around 0.01, scaled in the kernel.
	ll += Normal.Logp(-0.5, 1, x[s])

	// Priors of the renewal process
	// -----------------------------
	//  We allow the inputs to move somewhat.
	sigma := 1/math.Sqrt(1 + float64(len(x[i0:])))
	mu := - sigma*sigma/2
	ll += Normal.Logp(mu, sigma, x[t])
	ll += Normal.Logps(0, math.Exp(x[t]), x[i0:]...)

	return ll
}

// seasonable + periodic priors

type SARPriors struct {
}

func (m *SARPriors) NTheta() int {
	return 1
}

func (m *SARPriors) Observe(x []float64) float64 {
	const (
		c1 = iota // trend variance
		c2        // season variance
		l1        // trend length scale
		l2        // season length scale
		s         // noise variance
		t		  // warping
		i0        // first transformation (relative step change)
	)

	ll := 0.

	// Priors of the Gaussian process
	// ------------------------------
	// Variance is mostly less than 1.
	ll += Normal.Logp(-1, 1, x[c1])
	ll += Normal.Logp(-1, 1, x[c2])
	// Length scale is around 1, in wide margins.
	ll += Normal.Logp(-2, 2, x[l1])
	ll += Normal.Logp(-2, 2, x[l2])
	// Noise variance is around 0.01, scaled in the kernel.
	ll += Normal.Logp(-0.5, 1, x[s])

	// Priors of the renewal process
	// -----------------------------
	//  We allow the inputs to move somewhat.
	sigma := 1/math.Sqrt(1 + float64(len(x[i0:])))
	mu := - sigma*sigma/2
	ll += Normal.Logp(mu, sigma, x[t])
	ll += Normal.Logps(0, math.Exp(x[t]), x[i0:]...)

	return ll
}
